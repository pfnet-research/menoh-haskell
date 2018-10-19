{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

import qualified Codec.Picture as Picture
import qualified Codec.Picture.Types as Picture
import Control.Concurrent.Async
import Control.Exception
import Control.Monad
import qualified Data.ByteString as BS
import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Unboxed as VU
import Foreign
import System.FilePath

import Test.Tasty.HUnit
import Test.Tasty.TH

import Menoh
import Paths_menoh (getDataDir)

#include <menoh/version.h>

#define MIN_VERSION_libmenoh(major,minor,patch) (\
  (major) <  MENOH_MAJOR_VERSION || \
  (major) == MENOH_MAJOR_VERSION && (minor) <  MENOH_MINOR_VERSION || \
  (major) == MENOH_MAJOR_VERSION && (minor) == MENOH_MINOR_VERSION && (patch) <= MENOH_PATCH_VERSION)

------------------------------------------------------------------------

case_basicWriteBuffer_vector ::  Assertion
case_basicWriteBuffer_vector = do
  allocaArray 9 $ \(p :: Ptr Float) -> do
    basicWriteBuffer DTypeFloat [3,3] (castPtr p) (VG.tail (V.fromList xs))
    ys <- peekArray 9 p
    ys @?= (tail xs)
  where
    xs = [0..9]

case_basicWriteBuffer_vector_storable ::  Assertion
case_basicWriteBuffer_vector_storable = do
  allocaArray 9 $ \(p :: Ptr Float) -> do
    basicWriteBuffer DTypeFloat [3,3] (castPtr p) (VG.tail (VS.fromList xs))
    ys <- peekArray 9 p
    ys @?= tail xs
  where
    xs = [0..9]

case_basicWriteBuffer_vector_unboxed ::  Assertion
case_basicWriteBuffer_vector_unboxed = do
  allocaArray 9 $ \(p :: Ptr Float) -> do
    basicWriteBuffer DTypeFloat [3,3] (castPtr p) (VG.tail (VU.fromList xs))
    ys <- peekArray 9 p
    ys @?= tail xs
  where
    xs = [0..9]

case_basicWriteBuffer_list ::  Assertion
case_basicWriteBuffer_list = do
  allocaArray 9 $ \(p :: Ptr Float) -> do
    basicWriteBuffer DTypeFloat [3,3] (castPtr p) (map V.fromList xss)
    ys <- peekArray 9 p
    ys @?= concat xss
  where
    xss = [[1,2,3], [4,5,6], [7,8,9]]

------------------------------------------------------------------------

case_loading_nonexistent_model_file :: Assertion
case_loading_nonexistent_model_file = do
  dataDir <- getDataDir
  ret <- try $ makeModelDataFromONNX $ dataDir </> "data" </> "nonexistent_model.onnx"
  case ret of
    Left (ErrorInvalidFilename _msg) -> return ()
    _ -> assertFailure "should throw ErrorInvalidFilename"


case_empty_output :: Assertion
case_empty_output = do
  images <- loadMNISTImages
  let batch_size = length images

  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, [batch_size, mnist_channel_num, mnist_height, mnist_width])]
           []
           model_data
  optimizeModelData model_data vpt
  model <- makeModel vpt model_data "mkldnn"

  -- Run the model
  writeBuffer model mnist_in_name images
  run model

  -- but we cannot retrieve results
  return ()


case_insufficient_input :: Assertion
case_insufficient_input = do
  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  ret <- try $ makeVariableProfileTable
    []
    [(mnist_out_name, DTypeFloat)]
    model_data
  case ret of
    Left (ErrorVariableNotFound _msg) -> return ()
    _ -> assertFailure "should throw ErrorVariableNotFound"


case_bad_input :: Assertion
case_bad_input = do
  images <- loadMNISTImages

  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  ret <- try $ makeVariableProfileTable
           [ (mnist_in_name, DTypeFloat, [length images, mnist_channel_num, mnist_height, mnist_width])
           , ("bad input name", DTypeFloat, [1,8])
           ]
           [(mnist_out_name, DTypeFloat)]
           model_data
#if MIN_VERSION_libmenoh(1,1,0)
  case ret of
    Left (InputNotFoundError _msg) -> return ()
    _ -> assertFailure "should throw InputNotFoundError"
#else
  vpt <- case ret of
           Left (err :: Error) -> throwIO err
           Right vpt -> return vpt
  optimizeModelData model_data vpt
  model <- makeModel vpt model_data "mkldnn"

  -- Run the model
  writeBuffer model mnist_in_name images
  run model
  (vs :: [V.Vector Float]) <- readBuffer model mnist_out_name
  forM_ (zip [0..9] vs) $ \(i, scores) -> do
    V.maxIndex scores @?= i
#endif


case_bad_output :: Assertion
case_bad_output = do
  images <- loadMNISTImages

  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  ret <- try $ makeVariableProfileTable
    [(mnist_in_name, DTypeFloat, [length images, mnist_channel_num, mnist_height, mnist_width])]
    [(mnist_out_name, DTypeFloat), ("bad output name", DTypeFloat)]
    model_data
  case ret of
#if MIN_VERSION_libmenoh(1,1,0)
    Left (OutputNotFoundError _msg) -> return ()
    _ -> assertFailure "should throw OutputNotFoundError"
#else
    Left (ErrorVariableNotFound _msg) -> return ()
    _ -> assertFailure "should throw ErrorVariableNotFound"
#endif

------------------------------------------------------------------------

-- Aliases to onnx's node input and output tensor name
mnist_in_name, mnist_out_name :: String
mnist_in_name  = "139900320569040"
mnist_out_name = "139898462888656"

mnist_channel_num, mnist_height, mnist_width :: Int
mnist_channel_num = 1
mnist_height = 28
mnist_width  = 28

loadMNISTImages :: IO [VS.Vector Float]
loadMNISTImages = do
  dataDir <- getDataDir
  forM [(0::Int)..9] $ \i -> do
    ret <- Picture.readImage $ dataDir </> "data" </> (show i ++ ".png")
    case ret of
      Left e -> error e
      Right img -> return
        $ VG.map fromIntegral
        $ Picture.imageData
        $ Picture.extractLumaPlane
        $ Picture.convertRGB8
        $ img

loadMNISTModel :: Int -> IO Model
loadMNISTModel batch_size = do
  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, [batch_size, mnist_channel_num, mnist_height, mnist_width])]
           [(mnist_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt
  makeModel vpt model_data "mkldnn"

#if MIN_VERSION_libmenoh(1,1,0)
loadMNISTModelFromByteString :: Int -> IO Model
loadMNISTModelFromByteString batch_size = do
  dataDir <- getDataDir
  b <- BS.readFile $ dataDir </> "data" </> "mnist.onnx"
  model_data <- makeModelDataFromByteString b
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, [batch_size, mnist_channel_num, mnist_height, mnist_width])]
           [(mnist_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt
  makeModel vpt model_data "mkldnn"
#endif

case_MNIST :: Assertion
case_MNIST = do
  images <- loadMNISTImages
  model <- loadMNISTModel (length images)

  -- Run the model
  writeBuffer model mnist_in_name images
  run model
  (vs :: [V.Vector Float]) <- readBuffer model mnist_out_name
  forM_ (zip [0..9] vs) $ \(i, scores) -> do
    V.maxIndex scores @?= i

  -- Run the same model more than once, but with the different order
  writeBuffer model mnist_in_name (reverse images)
  run model
  (vs' :: [V.Vector Float]) <- readBuffer model mnist_out_name
  forM_ (zip [9,8..0] vs') $ \(i, scores) -> do
    V.maxIndex scores @?= i

case_MNIST_concurrently :: Assertion
case_MNIST_concurrently = do
  images <- loadMNISTImages
  let batch_size = length images

  dataDir <- getDataDir
  model_data <- makeModelDataFromONNX $ dataDir </> "data" </> "mnist.onnx"
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, [batch_size, mnist_channel_num, mnist_height, mnist_width])]
           [(mnist_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt
  models <- replicateM 10 $ makeModel vpt model_data "mkldnn"

  _ <- flip mapConcurrently models $ \model -> do
    replicateM_ 10 $ do
      writeBuffer model mnist_in_name images
      run model
      (vs :: [V.Vector Float]) <- readBuffer model mnist_out_name
      forM_ (zip [0..9] vs) $ \(i, scores) -> do
        V.maxIndex scores @?= i
  return ()

case_makeModelDataFromByteString :: Assertion
case_makeModelDataFromByteString = do
#if !MIN_VERSION_libmenoh(1,1,0)
  return () -- XXX
#else
  images <- loadMNISTImages
  model1 <- loadMNISTModel (length images)
  model2 <- loadMNISTModelFromByteString (length images)

  -- Run the model (1)
  writeBuffer model1 mnist_in_name images
  run model1
  (vs1 :: [V.Vector Float]) <- readBuffer model1 mnist_out_name

  -- Run the model (2)
  writeBuffer model2 mnist_in_name images
  run model2
  (vs2 :: [V.Vector Float]) <- readBuffer model2 mnist_out_name

  vs2 @?= vs1
#endif

case_makeModelData :: Assertion
case_makeModelData = do
#if !MIN_VERSION_libmenoh(1,1,0)
  return () -- XXX
#else
  md <- makeModelData
  withArray [1,2,3,4,5,6] $ \(p :: Ptr Float) ->
    addParamterFromPtr md "W" DTypeFloat [2,3] p
  withArray [7,8] $ \(p :: Ptr Float) ->
    addParamterFromPtr md "b" DTypeFloat [2] p
  addNewNode md "FC"
  addInputNameToCurrentNode md "input"
  addInputNameToCurrentNode md "W"
  addInputNameToCurrentNode md "b"
  addOutputNameToCurrentNode md "output"

  vpt <- makeVariableProfileTable
         [("input", DTypeFloat, [1, 3])]
         [("output", DTypeFloat)]
         md

  optimizeModelData md vpt
  m <- makeModel vpt md "mkldnn"

  writeBuffer m "input" $ [VS.fromList [1::Float,2,3]]
  run m
  [r] <- readBuffer m "output"

  r @?= VS.fromList [21::Float,40]
#endif

------------------------------------------------------------------------
-- Test harness

main :: IO ()
main = $(defaultMainGenerator)
