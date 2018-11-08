{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
-----------------------------------------------------------------------------
-- |
-- Module      :  Menoh
-- Copyright   :  Copyright (c) 2018 Preferred Networks, Inc.
-- License     :  MIT (see the file LICENSE)
--
-- Maintainer  :  Masahiro Sakai <sakai@preferred.jp>
-- Stability   :  experimental
-- Portability :  non-portable
--
-- Haskell binding for /Menoh/ DNN inference library.
--
-- = Basic usage
--
-- 1. Load computation graph from ONNX file using 'makeModelDataFromONNXFile'.
-- 2. Specify input variable type/dimentions (in particular batch size) and
--    which output variables you want to retrieve. This can be done by
--    constructing 'VariableProfileTable' using 'makeVariableProfileTable'.
-- 3. Optimize 'ModelData' with respect to your 'VariableProfileTable' by using
--    'optimizeModelData'.
-- 4. Construct a 'Model' using 'makeModel' or 'makeModelWithConfig'.
--    If you want to use custom buffers instead of internally allocated ones,
--    You need to use more low level 'ModelBuilder'.
-- 5. Load input data. This can be done conveniently using 'writeBuffer'.
--    There are also more low-level API such as 'unsafeGetBuffer' and 'withBuffer'.
-- 6. Run inference using 'run'.
-- 7. Retrieve the result data. This can be done conveniently using 'readBuffer'.
--
-- = Note on thread safety
--
-- TL;DR: If you want to use Menoh from multiple haskell threads, you need to
-- use /threaded/ RTS by supplying @-threaded@ option to GHC.
--
-- Menoh uses thread local storage (TLS) for storing error information, and
-- the only way to use TLS safely is to use in /bound/ threads
-- (see "Control.Concurrent#boundthreads").
--
-- * In /threaded RTS/ (i.e. 'rtsSupportsBoundThreads' is True), this module
--   runs computation in bound threads by using 'runInBoundThread'. (If the
--   calling thread is not bound, 'runInBoundThread' create a bound thread
--   temporarily and run the computation inside it).
--
-- * In /non-threaded RTS/, this module /does not/ use 'runInBoundThread' and
--   is therefore unsafe to use from multiple haskell threads. Using non-threaded
--   RTS is allowed for the sake of convenience (e.g. running in GHCi) despite
--   its unsafety.
--
-----------------------------------------------------------------------------

#include "MachDeps.h"
#include <menoh/version.h>

#define MIN_VERSION_libmenoh(major,minor,patch) (\
  (major) <  MENOH_MAJOR_VERSION || \
  (major) == MENOH_MAJOR_VERSION && (minor) <  MENOH_MINOR_VERSION || \
  (major) == MENOH_MAJOR_VERSION && (minor) == MENOH_MINOR_VERSION && (patch) <= MENOH_PATCH_VERSION)

module Menoh
  (
  -- * Basic data types
    Dims
  , DType (..)
  , Error (..)

  -- * ModelData type
  , ModelData (..)
  , makeModelDataFromONNXFile
  , makeModelDataFromONNX
  , makeModelDataFromONNXByteString
  , optimizeModelData
  -- ** Manual construction API
  , makeModelData
  , addParamterFromPtr
  , addNewNode
  , addInputNameToCurrentNode
  , addOutputNameToCurrentNode
  , AttributeType (..)
  , addAttribute

  -- * VariableProfileTable
  , VariableProfileTable (..)
  , makeVariableProfileTable
  , vptGetDType
  , vptGetDims

  -- * Model type
  , Model (..)
  , makeModel
  , makeModelWithConfig
  , run
  , getDType
  , getDims
  -- ** Accessors for buffers
  , ToBuffer (..)
  , FromBuffer (..)
  , writeBuffer
  , readBuffer
  -- ** Low-level accessors for buffers
  , unsafeGetBuffer
  , withBuffer
  -- ** Deprecated accessors for buffers
  , HasDType (..)
  , writeBufferFromVector
  , writeBufferFromStorableVector
  , readBufferToVector
  , readBufferToStorableVector

  -- * Misc
  , version
  , bindingVersion

  -- * Low-level API

  -- ** Builder for 'VariableProfileTable'
  , VariableProfileTableBuilder (..)
  , makeVariableProfileTableBuilder
  , addInputProfileDims2
  , addInputProfileDims4
  , addOutputName
  , addOutputProfile
  , AddOutput (..)
  , buildVariableProfileTable

  -- ** Builder for 'Model'
  , ModelBuilder (..)
  , makeModelBuilder
  , attachExternalBuffer
  , buildModel
  , buildModelWithConfig
  ) where

import Control.Applicative
import Control.Concurrent
import Control.Monad
import Control.Monad.Trans.Control (MonadBaseControl, liftBaseOp)
import Control.Monad.IO.Class
import Control.Exception
import qualified Data.Aeson as J
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Proxy
import Data.Typeable
import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Storable.Mutable as VSM
import qualified Data.Vector.Unboxed as VU
import Data.IntMap (IntMap)
import qualified Data.IntMap as IntMap
import Data.Version
import Foreign
import Foreign.C

import qualified Menoh.Base as Base
import qualified Paths_menoh

-- ------------------------------------------------------------------------

-- | Functions in this module can throw this exception type.
data Error
  = ErrorStdError String
  | ErrorUnknownError String
  | ErrorInvalidFilename String
  | ErrorONNXParseError String
  | ErrorInvalidDType String
  | ErrorInvalidAttributeType String
  | ErrorUnsupportedOperatorAttribute String
  | ErrorDimensionMismatch String
  | ErrorVariableNotFound String
  | ErrorIndexOutOfRange String
  | ErrorJSONParseError String
  | ErrorInvalidBackendName String
  | ErrorUnsupportedOperator String
  | ErrorFailedToConfigureOperator String
  | ErrorBackendError String
  | ErrorSameNamedVariableAlreadyExist String
  | UnsupportedInputDims String
  | SameNamedParameterAlreadyExist String
  | SameNamedAttributeAlreadyExist String
  | InvalidBackendConfigError String
  | InputNotFoundError String
  | OutputNotFoundError String
  deriving (Eq, Ord, Show, Read, Typeable)

instance Exception Error

runMenoh :: IO Base.MenohErrorCode -> IO ()
runMenoh m = runInBoundThread' $ do
  e <- m
  if e == Base.menohErrorCodeSuccess then
    return ()
  else do
    s <- peekCString =<< Base.menoh_get_last_error_message
    case IntMap.lookup (fromIntegral e) table of
      Just ex -> throwIO $ ex s
      Nothing -> throwIO $ ErrorUnknownError $ s ++ "(error code: " ++ show (fromIntegral e :: Int) ++ ")"
  where
    table :: IntMap (String -> Error)
    table = IntMap.fromList $ map (\(k,v) -> (fromIntegral k, v)) $
      [ (Base.menohErrorCodeStdError                      , ErrorStdError)
      , (Base.menohErrorCodeUnknownError                  , ErrorUnknownError)
      , (Base.menohErrorCodeInvalidFilename               , ErrorInvalidFilename)
      , (Base.menohErrorCodeOnnxParseError                , ErrorONNXParseError)
      , (Base.menohErrorCodeInvalidDtype                  , ErrorInvalidDType)
      , (Base.menohErrorCodeInvalidAttributeType          , ErrorInvalidAttributeType)
      , (Base.menohErrorCodeUnsupportedOperatorAttribute  , ErrorUnsupportedOperatorAttribute)
      , (Base.menohErrorCodeDimensionMismatch             , ErrorDimensionMismatch)
      , (Base.menohErrorCodeVariableNotFound              , ErrorVariableNotFound)
      , (Base.menohErrorCodeIndexOutOfRange               , ErrorIndexOutOfRange)
      , (Base.menohErrorCodeJsonParseError                , ErrorJSONParseError)
      , (Base.menohErrorCodeInvalidBackendName            , ErrorInvalidBackendName)
      , (Base.menohErrorCodeUnsupportedOperator           , ErrorUnsupportedOperator)
      , (Base.menohErrorCodeFailedToConfigureOperator     , ErrorFailedToConfigureOperator)
      , (Base.menohErrorCodeBackendError                  , ErrorBackendError)
      , (Base.menohErrorCodeSameNamedVariableAlreadyExist , ErrorSameNamedVariableAlreadyExist)
      , (Base.menohErrorCodeUnsupportedInputDims          , UnsupportedInputDims)
      , (Base.menohErrorCodeSameNamedParameterAlreadyExist, SameNamedParameterAlreadyExist)
      , (Base.menohErrorCodeSameNamedAttributeAlreadyExist, SameNamedAttributeAlreadyExist)
      , (Base.menohErrorCodeInvalidBackendConfigError     , InvalidBackendConfigError)
      , (Base.menohErrorCodeInputNotFoundError            , InputNotFoundError)
      , (Base.menohErrorCodeOutputNotFoundError           , OutputNotFoundError)
      ]

runInBoundThread' :: IO a -> IO a
runInBoundThread' action
  | rtsSupportsBoundThreads = runInBoundThread action
  | otherwise = action

-- ------------------------------------------------------------------------

-- | Data type of array elements
data DType
  = DTypeFloat                    -- ^ single precision floating point number
  | DTypeUnknown !Base.MenohDType -- ^ types that this binding is unware of
  deriving (Eq, Ord, Show, Read)

instance Enum DType where
  toEnum x
    | x == fromIntegral Base.menohDtypeFloat = DTypeFloat
    | otherwise = DTypeUnknown (fromIntegral x)

  fromEnum DTypeFloat = fromIntegral Base.menohDtypeFloat
  fromEnum (DTypeUnknown i) = fromIntegral i

dtypeSize :: DType -> Int
dtypeSize DTypeFloat = sizeOf (undefined :: CFloat)
dtypeSize (DTypeUnknown _) = error "Menoh.dtypeSize: unknown DType"

{-# DEPRECATED HasDType "use FromBuffer/ToBuffer instead" #-}
-- | Haskell types that have associated 'DType' type code.
class Storable a => HasDType a where
  dtypeOf :: Proxy a -> DType

instance HasDType CFloat where
  dtypeOf _ = DTypeFloat

#if SIZEOF_HSFLOAT == SIZEOF_FLOAT

instance HasDType Float where
  dtypeOf _ = DTypeFloat

#endif

-- ------------------------------------------------------------------------

-- | Dimensions of array
type Dims = [Int]

-- ------------------------------------------------------------------------

-- | @ModelData@ contains model parameters and computation graph structure.
newtype ModelData = ModelData (ForeignPtr Base.MenohModelData)

{-# DEPRECATED makeModelDataFromONNX "use makeModelDataFromONNXFile instead" #-}
-- | Load onnx file and make 'ModelData'.
makeModelDataFromONNX :: MonadIO m => FilePath -> m ModelData
makeModelDataFromONNX = makeModelDataFromONNXFile

-- | Load onnx file and make 'ModelData'.
makeModelDataFromONNXFile :: MonadIO m => FilePath -> m ModelData
makeModelDataFromONNXFile fpath = liftIO $ withCString fpath $ \fpath' -> alloca $ \ret -> do
  runMenoh $ Base.menoh_make_model_data_from_onnx fpath' ret
  liftM ModelData $ newForeignPtr Base.menoh_delete_model_data_funptr =<< peek ret

-- | make 'ModelData' from on-memory 'BS.ByteString'.
makeModelDataFromONNXByteString :: MonadIO m => BS.ByteString -> m ModelData
makeModelDataFromONNXByteString b = liftIO $ BS.useAsCStringLen b $ \(p,len) -> alloca $ \ret -> do  
  runMenoh $ Base.menoh_make_model_data_from_onnx_data_on_memory p (fromIntegral len) ret
  liftM ModelData $ newForeignPtr Base.menoh_delete_model_data_funptr =<< peek ret

-- | Optimize function for 'ModelData'.
--
-- This function modify given 'ModelData'.
optimizeModelData :: MonadIO m => ModelData -> VariableProfileTable -> m ()
optimizeModelData (ModelData m) (VariableProfileTable vpt) = liftIO $
  withForeignPtr m $ \m' -> withForeignPtr vpt $ \vpt' ->
    runMenoh $ Base.menoh_model_data_optimize m' vpt'

-- | Make empty model_data
makeModelData :: MonadIO m => m ModelData
makeModelData = liftIO $ alloca $ \ret -> do
  runMenoh $ Base.menoh_make_model_data ret
  liftM ModelData $ newForeignPtr Base.menoh_delete_model_data_funptr =<< peek ret

-- | Add a new parameter in model_data
--
-- Duplication of parameter_name is not allowed and it throws error.
addParamterFromPtr :: MonadIO m => ModelData -> String -> DType -> Dims -> Ptr a -> m ()
addParamterFromPtr (ModelData m) name dtype dims p = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' -> withArrayLen (map fromIntegral dims) $ \n dims' ->
    runMenoh $ Base.menoh_model_data_add_parameter m' name' (fromIntegral (fromEnum dtype)) (fromIntegral n) dims' p

-- | Add a new node to model_data
addNewNode :: MonadIO m => ModelData -> String -> m ()
addNewNode (ModelData m) name = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_model_data_add_new_node m' name'

-- | Add a new input name to latest added node in model_data
addInputNameToCurrentNode :: MonadIO m => ModelData -> String -> m ()
addInputNameToCurrentNode (ModelData m) name = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_model_data_add_input_name_to_current_node m' name'

-- | Add a new output name to latest added node in model_data
addOutputNameToCurrentNode :: MonadIO m => ModelData -> String -> m ()
addOutputNameToCurrentNode (ModelData m) name = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_model_data_add_output_name_to_current_node m' name'

-- | A class of types that can be added to nodes using 'addAttribute'.
class AttributeType value where
  basicAddAttribute :: Ptr Base.MenohModelData -> CString -> value -> IO ()

instance AttributeType Int where
  basicAddAttribute m' name' value =
    runMenoh $ Base.menoh_model_data_add_attribute_int_to_current_node m' name' (fromIntegral value)

instance AttributeType Float where
  basicAddAttribute m' name' value =
    runMenoh $ Base.menoh_model_data_add_attribute_float_to_current_node m' name' (realToFrac value)

instance AttributeType [Int] where
  basicAddAttribute m' name' values =
    withArrayLen (map fromIntegral values) $ \n values' ->
      runMenoh $ Base.menoh_model_data_add_attribute_ints_to_current_node m' name' (fromIntegral n) values'

instance AttributeType [Float] where
  basicAddAttribute m' name' values =
    withArrayLen (map realToFrac values) $ \n values' ->
      runMenoh $ Base.menoh_model_data_add_attribute_floats_to_current_node m' name' (fromIntegral n) values'

-- | Add a new attribute to latest added node in model_data
addAttribute :: (AttributeType value, MonadIO m) => ModelData -> String -> value -> m ()
addAttribute (ModelData m) name value = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' ->
    basicAddAttribute m' name' value

-- ------------------------------------------------------------------------

-- | Builder for creation of 'VariableProfileTable'.
newtype VariableProfileTableBuilder
  = VariableProfileTableBuilder (ForeignPtr Base.MenohVariableProfileTableBuilder)

-- | Factory function for 'VariableProfileTableBuilder'.
makeVariableProfileTableBuilder :: MonadIO m => m VariableProfileTableBuilder
makeVariableProfileTableBuilder = liftIO $ alloca $ \p -> do
  runMenoh $ Base.menoh_make_variable_profile_table_builder p
  liftM VariableProfileTableBuilder $ newForeignPtr Base.menoh_delete_variable_profile_table_builder_funptr =<< peek p

addInputProfileDims :: MonadIO m => VariableProfileTableBuilder -> String -> DType -> Dims -> m ()
addInputProfileDims (VariableProfileTableBuilder vpt) name dtype dims =
  liftIO $
    withForeignPtr vpt $ \vpt' -> withCString name $ \name' -> withArrayLen (map fromIntegral dims) $ \n dims' ->
      runMenoh $ Base.menoh_variable_profile_table_builder_add_input_profile
        vpt' name' (fromIntegral (fromEnum dtype)) (fromIntegral n) dims'

-- | Add 2D input profile.
--
-- Input profile contains name, dtype and dims @(num, size)@.
-- This 2D input is conventional batched 1D inputs.
{-# DEPRECATED addInputProfileDims2 "use addInputProfileDims instead" #-}
addInputProfileDims2
  :: MonadIO m
  => VariableProfileTableBuilder
  -> String
  -> DType
  -> (Int, Int) -- ^ (num, size)
  -> m ()
addInputProfileDims2 (VariableProfileTableBuilder vpt) name dtype (num, size) = liftIO $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_variable_profile_table_builder_add_input_profile_dims_2
      vpt' name' (fromIntegral (fromEnum dtype))
      (fromIntegral num) (fromIntegral size)

-- | Add 4D input profile
--
-- Input profile contains name, dtype and dims @(num, channel, height, width)@.
-- This 4D input is conventional batched image inputs. Image input is
-- 3D (channel, height, width).
{-# DEPRECATED addInputProfileDims4 "use addInputProfileDims instead" #-}
addInputProfileDims4
  :: MonadIO m
  => VariableProfileTableBuilder
  -> String
  -> DType
  -> (Int, Int, Int, Int) -- ^ (num, channel, height, width)
  -> m ()
addInputProfileDims4 (VariableProfileTableBuilder vpt) name dtype (num, channel, height, width) = liftIO $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_variable_profile_table_builder_add_input_profile_dims_4
      vpt' name' (fromIntegral (fromEnum dtype))
      (fromIntegral num) (fromIntegral channel) (fromIntegral height) (fromIntegral width)

-- | Add output name
--
-- Output profile contains name and dtype. Its 'Dims' and 'DType' are calculated
-- automatically, so that you don't need to specify explicitly.
addOutputName :: MonadIO m => VariableProfileTableBuilder -> String -> m ()
addOutputName (VariableProfileTableBuilder vpt) name = liftIO $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_variable_profile_table_builder_add_output_name
      vpt' name'

{-# DEPRECATED addOutputProfile "use addOutputName instead" #-}
-- | Add output profile
--
-- Output profile contains name and dtype. Its 'Dims' are calculated automatically,
-- so that you don't need to specify explicitly.
addOutputProfile :: MonadIO m => VariableProfileTableBuilder -> String -> DType -> m ()
addOutputProfile (VariableProfileTableBuilder vpt) name dtype = liftIO $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_variable_profile_table_builder_add_output_profile
      vpt' name' (fromIntegral (fromEnum dtype))

-- | Type class for abstracting 'addOutputProfile' and 'addOutputName'.
class AddOutput a where
  addOutput :: VariableProfileTableBuilder -> a -> IO ()

instance AddOutput String where
  addOutput = addOutputName

instance AddOutput (String, DType) where
  addOutput b (name,_dtype) = addOutputName b name

-- | Factory function for 'VariableProfileTable'
buildVariableProfileTable
  :: MonadIO m
  => VariableProfileTableBuilder
  -> ModelData
  -> m VariableProfileTable
buildVariableProfileTable (VariableProfileTableBuilder b) (ModelData m) = liftIO $
  withForeignPtr b $ \b' -> withForeignPtr m $ \m' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_build_variable_profile_table b' m' ret
    liftM VariableProfileTable $ newForeignPtr Base.menoh_delete_variable_profile_table_funptr =<< peek ret

-- ------------------------------------------------------------------------

-- | @VariableProfileTable@ contains information of dtype and dims of variables.
--
-- Users can access to dtype and dims via 'vptGetDType' and 'vptGetDims'.
newtype VariableProfileTable
  = VariableProfileTable (ForeignPtr Base.MenohVariableProfileTable)

-- | Convenient function for constructing 'VariableProfileTable'.
--
-- If you need finer control, you can use 'VariableProfileTableBuidler'.
makeVariableProfileTable
  :: (AddOutput a, MonadIO m)
  => [(String, DType, Dims)]  -- ^ input names with dtypes and dims
  -> [a]                      -- ^ required output informations (@`String`@ or @('String', 'DType')@)
  -> ModelData                -- ^ model data
  -> m VariableProfileTable
makeVariableProfileTable input_name_and_dims_pair_list required_output_name_list model_data = liftIO $ runInBoundThread' $ do
  b <- makeVariableProfileTableBuilder
  forM_ input_name_and_dims_pair_list $ \(name,dtype,dims) -> do
    addInputProfileDims b name dtype dims
  mapM_ (addOutput b) required_output_name_list
  buildVariableProfileTable b model_data

-- | Accessor function for 'VariableProfileTable'
--
-- Select variable name and get its 'DType'.
vptGetDType :: MonadIO m => VariableProfileTable -> String -> m DType
vptGetDType (VariableProfileTable vpt) name = liftIO $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_variable_profile_table_get_dims_size vpt' name' ret
    (toEnum . fromIntegral) <$> peek ret

-- | Accessor function for 'VariableProfileTable'
--
-- Select variable name and get its 'Dims'.
vptGetDims :: MonadIO m => VariableProfileTable -> String -> m Dims
vptGetDims (VariableProfileTable vpt) name = liftIO $ runInBoundThread' $
  withForeignPtr vpt $ \vpt' -> withCString name $ \name' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_variable_profile_table_get_dims_size vpt' name' ret
    size <- peek ret
    forM [0..size-1] $ \i -> do
      runMenoh $ Base.menoh_variable_profile_table_get_dims_at vpt' name' (fromIntegral i) ret
      fromIntegral <$> peek ret

-- ------------------------------------------------------------------------

-- | Helper for creating of 'Model'.
newtype ModelBuilder = ModelBuilder (ForeignPtr Base.MenohModelBuilder)

-- | Factory function for 'ModelBuilder'
makeModelBuilder :: MonadIO m => VariableProfileTable -> m ModelBuilder
makeModelBuilder (VariableProfileTable vpt) = liftIO $
  withForeignPtr vpt $ \vpt' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_make_model_builder vpt' ret
    liftM ModelBuilder $ newForeignPtr Base.menoh_delete_model_builder_funptr =<< peek ret

-- | Attach a buffer which allocated by users.
--
-- Users can attach a external buffer which they allocated to target variable.
--
-- Variables attached no external buffer are attached internal buffers allocated
-- automatically.
--
-- Users can get that internal buffer handle by calling 'unsafeGetBuffer' etc. later.
attachExternalBuffer :: MonadIO m => ModelBuilder -> String -> Ptr a -> m ()
attachExternalBuffer (ModelBuilder m) name buf = liftIO $
  withForeignPtr m $ \m' -> withCString name $ \name' ->
    runMenoh $ Base.menoh_model_builder_attach_external_buffer m' name' buf

-- | Factory function for 'Model'.
buildModel
  :: MonadIO m
  => ModelBuilder
  -> ModelData
  -> String  -- ^ backend name
  -> m Model
buildModel builder m backend = liftIO $
  withCString "" $
    buildModelWithConfigString builder m backend

-- | Similar to 'buildModel', but backend specific configuration can be supplied as JSON.
buildModelWithConfig
  :: (MonadIO m, J.ToJSON a)
  => ModelBuilder
  -> ModelData
  -> String  -- ^ backend name
  -> a       -- ^ backend config
  -> m Model
buildModelWithConfig builder m backend backend_config = liftIO $
  BS.useAsCString (BL.toStrict (J.encode backend_config)) $
    buildModelWithConfigString builder m backend

buildModelWithConfigString
  :: MonadIO m
  => ModelBuilder
  -> ModelData
  -> String  -- ^ backend name
  -> CString -- ^ backend config
  -> m Model
buildModelWithConfigString (ModelBuilder builder) (ModelData m) backend backend_config = liftIO $
  withForeignPtr builder $ \builder' -> withForeignPtr m $ \m' -> withCString backend $ \backend' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_build_model builder' m' backend' backend_config ret
    liftM Model $ newForeignPtr Base.menoh_delete_model_funptr =<< peek ret

-- ------------------------------------------------------------------------

-- | ONNX model with input/output buffers
newtype Model = Model (ForeignPtr Base.MenohModel)

-- | Run model inference.
--
-- This function can't be called asynchronously.
run :: MonadIO m => Model -> m ()
run (Model model) = liftIO $ withForeignPtr model $ \model' -> do
  runMenoh $ Base.menoh_model_run model'

-- | Get 'DType' of target variable.
getDType :: MonadIO m => Model -> String -> m DType
getDType (Model m) name = liftIO $ do
  withForeignPtr m $ \m' -> withCString name $ \name' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_model_get_variable_dtype m' name' ret
    liftM (toEnum . fromIntegral) $ peek ret

-- | Get 'Dims' of target variable.
getDims :: MonadIO m => Model -> String -> m Dims
getDims (Model m) name = liftIO $ runInBoundThread' $ do
  withForeignPtr m $ \m' -> withCString name $ \name' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_model_get_variable_dims_size m' name' ret
    size <- peek ret
    forM [0..size-1] $ \i -> do
      runMenoh $ Base.menoh_model_get_variable_dims_at m' name' (fromIntegral i) ret
      fromIntegral <$> peek ret

-- ------------------------------------------------------------------------
-- Accessing buffers

-- | Get a buffer handle attached to target variable.
--
-- Users can get a buffer handle attached to target variable.
-- If that buffer is allocated by users and attached to the variable by calling
-- 'attachExternalBuffer', returned buffer handle is same to it.
--
-- This function is unsafe because it does not prevent the model to be GC'ed and
-- the returned pointer become dangling pointer.
--
-- See also 'withBuffer'.
unsafeGetBuffer :: MonadIO m => Model -> String -> m (Ptr a)
unsafeGetBuffer (Model m) name = liftIO $ do
  withForeignPtr m $ \m' -> withCString name $ \name' -> alloca $ \ret -> do
    runMenoh $ Base.menoh_model_get_variable_buffer_handle m' name' ret
    peek ret

-- | This function takes a function which is applied to the buffer associated to specified variable.
-- The resulting action is then executed. The buffer is kept alive at least during the whole action,
-- even if it is not used directly inside.
-- Note that it is not safe to return the pointer from the action and use it after the action completes.
--
-- See also 'unsafeGetBuffer'.
withBuffer :: forall m r a. (MonadIO m, MonadBaseControl IO m) => Model -> String -> (Ptr a -> m r) -> m r
withBuffer (Model m) name f =
  liftBaseOp (withForeignPtr m) $ \m' ->
  (liftBaseOp (withCString name) ::  (CString -> m r) -> m r) $ \name' ->
  liftBaseOp alloca $ \ret -> do
    p <- liftIO $ do
      runMenoh $ Base.menoh_model_get_variable_buffer_handle m' name' ret
      peek ret
    f p

-- | Type that can be written to menoh's buffer.
class ToBuffer a where
  -- Basic method for implementing @ToBuffer@ class.
  -- Normal user should use 'writeBuffer' instead.
  basicWriteBuffer :: DType -> Dims -> Ptr () -> a -> IO ()

-- | Type that can be read from menoh's buffer.
class FromBuffer a where
  -- Basic method for implementing @FromBuffer@ class.
  -- Normal user should use 'readBuffer' instead.
  basicReadBuffer :: DType -> Dims -> Ptr () -> IO a

-- | Read values from the given model's buffer
readBuffer :: (FromBuffer a, MonadIO m) => Model -> String -> m a
readBuffer model name = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicReadBuffer dtype dims p

-- | Write values to the given model's buffer
writeBuffer :: (ToBuffer a, MonadIO m) => Model -> String -> a -> m ()
writeBuffer model name a = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicWriteBuffer dtype dims p a


-- | Default implementation of 'basicWriteBuffer' for 'VG.Vector' class
-- for the cases whete the 'Storable' is compatible for representation in buffers.
basicWriteBufferGenericVectorStorable
  :: forall v a. (VG.Vector v a, Storable a)
  => DType -> DType -> Dims -> Ptr () -> v a -> IO ()
basicWriteBufferGenericVectorStorable dtype0 dtype dims p vec = do
  let n = product dims
      p' = castPtr p
  checkDTypeAndSize "Menoh.basicWriteBufferGenericVectorStorable" (dtype, n) (dtype0, VG.length vec)
  forM_ [0..n-1] $ \i -> do
    pokeElemOff p' i (vec VG.! i)

-- | Default implementation of 'basicReadToBuffer' for 'VG.Vector' class
-- for the cases whete the 'Storable' is compatible for representation in buffers.
basicReadBufferGenericVectorStorable
  :: forall v a. (VG.Vector v a, Storable a)
  => DType -> DType -> Dims -> Ptr () -> IO (v a)
basicReadBufferGenericVectorStorable dtype0 dtype dims p = do
  checkDType "Menoh.basicReadBufferGenericVectorStorable" dtype dtype0
  let n = product dims
      p' = castPtr p
  VG.generateM n $ peekElemOff p'


-- | Default implementation of 'basicWriteBuffer' for 'VS.Vector' class
-- for the cases whete the 'Storable' is compatible for representation in buffers.
basicWriteBufferStorableVector
  :: forall a. (Storable a)
  => DType -> DType -> Dims -> Ptr () -> VS.Vector a -> IO ()
basicWriteBufferStorableVector dtype0 dtype dims p vec = do
  let n = product dims
  checkDTypeAndSize "Menoh.basicWriteBufferStorableVector" (dtype, n) (dtype0, VG.length vec)
  VS.unsafeWith vec $ \src -> do
    copyArray (castPtr p) src n

-- | Default implementation of 'basicReadToBuffer' for 'VS.Vector' class
-- for the cases whete the 'Storable' is compatible for representation in buffers.
basicReadBufferStorableVector
  :: forall a. (Storable a)
  => DType -> DType -> Dims -> Ptr () -> IO (VS.Vector a)
basicReadBufferStorableVector dtype0 dtype dims p = do
  checkDType "Menoh.basicReadBufferStorableVector" dtype dtype0
  let n = product dims
  vec <- VSM.new n
  VSM.unsafeWith vec $ \dst -> copyArray dst (castPtr p) n
  VS.unsafeFreeze vec


instance ToBuffer (V.Vector Float) where
  basicWriteBuffer = basicWriteBufferGenericVectorStorable DTypeFloat
instance FromBuffer (V.Vector Float) where
  basicReadBuffer = basicReadBufferGenericVectorStorable DTypeFloat


instance ToBuffer (VU.Vector Float) where
  basicWriteBuffer = basicWriteBufferGenericVectorStorable DTypeFloat
instance FromBuffer (VU.Vector Float) where
  basicReadBuffer = basicReadBufferGenericVectorStorable DTypeFloat


instance ToBuffer (VS.Vector Float) where
  basicWriteBuffer = basicWriteBufferStorableVector DTypeFloat
instance FromBuffer (VS.Vector Float) where
  basicReadBuffer = basicReadBufferStorableVector DTypeFloat


instance ToBuffer a => ToBuffer [a] where
  basicWriteBuffer _dtype [] _p _xs =
    throwIO $ ErrorDimensionMismatch $ "ToBuffer{[a]}.basicWriteBuffer: empty dims"
  basicWriteBuffer dtype (dim : dims) p xs = do
    unless (dim == length xs) $ do
      throwIO $ ErrorDimensionMismatch $ "ToBuffer{[a]}.basicWriteBuffer: dimension mismatch"
    let s = product dims * dtypeSize dtype
    forM_ (zip [0,s..] xs) $ \(offset,x) -> do
      basicWriteBuffer dtype dims (p `plusPtr` offset) x

instance FromBuffer a => FromBuffer [a] where
  basicReadBuffer _dtype [] _p =
    throwIO $ ErrorDimensionMismatch $ "FromBuffer{[a]}.basicReadBuffer: empty dims"
  basicReadBuffer dtype (dim : dims) p = do
    let s = product dims * dtypeSize dtype
    forM [0..dim-1] $ \i -> do
      basicReadBuffer dtype dims (p `plusPtr` (i*s))


checkDType :: String -> DType -> DType -> IO ()
checkDType name dtype1 dtype2
  | dtype1 /= dtype2 = throwIO $ ErrorInvalidDType $ name ++ ": dtype mismatch"
  | otherwise        = return ()

checkDTypeAndSize :: String -> (DType,Int) -> (DType,Int) -> IO ()
checkDTypeAndSize name (dtype1,n1) (dtype2,n2)
  | dtype1 /= dtype2 = throwIO $ ErrorInvalidDType $ name ++ ": dtype mismatch"
  | n1 /= n2         = throwIO $ ErrorDimensionMismatch $ name ++ ": dimension mismatch"
  | otherwise        = return ()


{-# DEPRECATED writeBufferFromVector, writeBufferFromStorableVector "Use ToBuffer class and writeBuffer instead" #-}

-- | Copy whole elements of 'VG.Vector' into a model's buffer
writeBufferFromVector :: forall v a m. (VG.Vector v a, HasDType a, MonadIO m) => Model -> String -> v a -> m ()
writeBufferFromVector model name vec = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicWriteBufferGenericVectorStorable (dtypeOf (Proxy :: Proxy a)) dtype dims p vec

-- | Copy whole elements of @'VS.Vector' a@ into a model's buffer
writeBufferFromStorableVector :: forall a m. (HasDType a, MonadIO m) => Model -> String -> VS.Vector a -> m ()
writeBufferFromStorableVector model name vec = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicWriteBufferStorableVector (dtypeOf (Proxy :: Proxy a)) dtype dims p vec

{-# DEPRECATED readBufferToVector, readBufferToStorableVector "Use FromBuffer class and readBuffer instead" #-}

-- | Read whole elements of 'Array' and return as a 'VG.Vector'.
readBufferToVector :: forall v a m. (VG.Vector v a, HasDType a, MonadIO m) => Model -> String -> m (v a)
readBufferToVector model name = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicReadBufferGenericVectorStorable (dtypeOf (Proxy :: Proxy a)) dtype dims p

-- | Read whole eleemnts of 'Array' and return as a 'VS.Vector'.
readBufferToStorableVector :: forall a m. (HasDType a, MonadIO m) => Model -> String -> m (VS.Vector a)
readBufferToStorableVector model name = liftIO $ withBuffer model name $ \p -> do
  dtype <- getDType model name
  dims <- getDims model name
  basicReadBufferStorableVector (dtypeOf (Proxy :: Proxy a)) dtype dims p

-- ------------------------------------------------------------------------

-- | Convenient methods for constructing  a 'Model'.
makeModel
  :: MonadIO m
  => VariableProfileTable    -- ^ variable profile table
  -> ModelData               -- ^ model data
  -> String                  -- ^ backend name
  -> m Model
makeModel vpt model_data backend_name = liftIO $ do
  b <- makeModelBuilder vpt
  buildModel b model_data backend_name

-- | Similar to 'makeModel' but backend-specific configuration can be supplied.
makeModelWithConfig
  :: (MonadIO m, J.ToJSON a)
  => VariableProfileTable    -- ^ variable profile table
  -> ModelData               -- ^ model data
  -> String                  -- ^ backend name
  -> a                       -- ^ backend config
  -> m Model
makeModelWithConfig vpt model_data backend_name backend_config = liftIO $ do
  b <- makeModelBuilder vpt
  buildModelWithConfig b model_data backend_name backend_config

-- ------------------------------------------------------------------------

-- | Menoh version which was supplied on compilation time via CPP macro.
version :: Version
#if MIN_VERSION_base(4,8,0)
version = makeVersion [Base.menoh_major_version, Base.menoh_minor_version, Base.menoh_patch_version]
#else
version = Version [Base.menoh_major_version, Base.menoh_minor_version, Base.menoh_patch_version] []
#endif

-- | Version of this Haskell binding. (Not the version of /Menoh/ itself)
bindingVersion :: Version
bindingVersion = Paths_menoh.version
