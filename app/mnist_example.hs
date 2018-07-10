{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import qualified Codec.Picture as Picture
import Control.Applicative
import Control.Monad
import Data.Monoid
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import Data.Version
import Options.Applicative
import Menoh
import System.FilePath
import Text.Printf

import Paths_menoh (getDataDir)

main :: IO ()
main = do
  putStrLn "mnist example"
  dataDir <- getDataDir
  opt <- execParser (parserInfo (dataDir </> "data"))

  let input_dir = optInputPath opt
      image_filenames =
        [ "0.png"
        , "1.png"
        , "2.png"
        , "3.png"
        , "4.png"
        , "5.png"
        , "6.png"
        , "7.png"
        , "8.png"
        , "9.png"
        ]
      batch_size  = length image_filenames
      channel_num = 1
      height = 28
      width  = 28
      category_num = 10
      input_dims, output_dims :: Dims
      input_dims  = [batch_size, channel_num, height, width]
      output_dims = [batch_size, category_num]

  images <- forM image_filenames $ \fname -> do
    ret <- Picture.readImage $ input_dir </> fname
    case ret of
      Left e -> error e
      Right img -> return $ convert width height img

  -- Aliases to onnx's node input and output tensor name
  let mnist_in_name  = "139900320569040"
      mnist_out_name = "139898462888656"

  -- Load ONNX model data
  model_data <- makeModelDataFromONNX (optModelPath opt)

  -- Specify inputs and outputs
  vpt <- makeVariableProfileTable
           [(mnist_in_name, DTypeFloat, input_dims)]
           [(mnist_out_name, DTypeFloat)]
           model_data
  optimizeModelData model_data vpt

  -- Construct computation primitive list and memories
  model <- makeModel vpt model_data "mkldnn"

  -- Copy input image data to model's input array
  writeBuffer model mnist_in_name images

  -- Run inference
  run model

  -- Get output
  (vs :: [V.Vector Float]) <- readBuffer model mnist_out_name
  forM_ (zip vs image_filenames) $ \(scores,fname) -> do
    let j = V.maxIndex scores
        s = scores V.! j
    printf "%s = %d : %f\n" fname j s

-- -------------------------------------------------------------------------

data Options
  = Options
  { optInputPath :: FilePath
  , optModelPath :: FilePath
  }

optionsParser :: FilePath -> Parser Options
optionsParser dataDir = Options
  <$> inputPathOption
  <*> modelPathOption
  where
    inputPathOption = strOption
      $  long "input"
      <> short 'i'
      <> metavar "DIR"
      <> help "input image path"
      <> value dataDir
      <> showDefault
    modelPathOption = strOption
      $  long "model"
      <> short 'm'
      <> metavar "PATH"
      <> help "onnx model path"
      <> value (dataDir </> "mnist.onnx")
      <> showDefault

parserInfo :: FilePath -> ParserInfo Options
parserInfo dir = info (helper <*> versionOption <*> optionsParser dir)
  $  fullDesc
  <> header "mnist_example - an example program of Menoh haskell binding"
  where
    versionOption :: Parser (a -> a)
    versionOption = infoOption (showVersion version)
      $  hidden
      <> long "version"
      <> help "Show version"

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

-- -------------------------------------------------------------------------
