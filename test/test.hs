{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

import qualified Codec.Picture as Picture
import Control.Concurrent.Async
import Control.Monad
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
      Right img -> return $ convert mnist_width mnist_height img

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

-- -------------------------------------------------------------------------

convert :: Int -> Int -> Picture.DynamicImage -> VS.Vector Float
convert w h = reorderToNCHW . resize (w,h) . crop . Picture.convertRGB8

crop :: Picture.Pixel a => Picture.Image a -> Picture.Image a
crop img = Picture.generateImage (\x y -> Picture.pixelAt img (base_x + x) (base_y + y)) shortEdge shortEdge
  where
    shortEdge = min (Picture.imageWidth img) (Picture.imageHeight img)
    base_x = (Picture.imageWidth  img - shortEdge) `div` 2
    base_y = (Picture.imageHeight img - shortEdge) `div` 2

-- TODO: Should we do some kind of interpolation?
resize :: Picture.Pixel a => (Int,Int) -> Picture.Image a -> Picture.Image a
resize (w,h) img = Picture.generateImage (\x y -> Picture.pixelAt img (x * orig_w `div` w) (y * orig_h `div` h)) w h
  where
    orig_w = Picture.imageWidth  img
    orig_h = Picture.imageHeight img

reorderToNCHW :: Picture.Image Picture.PixelRGB8 -> VS.Vector Float
reorderToNCHW img = VS.generate (Picture.imageHeight img * Picture.imageWidth img) f
  where
    f i =
      case Picture.pixelAt img x y of
        Picture.PixelRGB8 r g b ->
          (fromIntegral r + fromIntegral g + fromIntegral b) / 3
      where
        (y,x) = i `divMod` Picture.imageWidth img

------------------------------------------------------------------------
-- Test harness

main :: IO ()
main = $(defaultMainGenerator)
