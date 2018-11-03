{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import qualified Codec.Picture as Picture
import Control.Applicative
import Control.Monad
import Data.List
import Data.Monoid
import Data.Ord
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import Data.Version
import Options.Applicative
import Menoh
import Text.Printf

main :: IO ()
main = do
  putStrLn "vgg16 example"
  opt <- execParser parserInfo

  let batch_size  = 1
      channel_num = 3
      height = 224
      width  = 224
      category_num = 1000
      input_dims, output_dims :: Dims
      input_dims  = [batch_size, channel_num, height, width]
      output_dims = [batch_size, category_num]

  ret <- Picture.readImage (optInputImagePath opt)
  let image_data =
        case ret of
          Left e -> error e
          Right img -> convert width height img

  -- Aliases to onnx's node input and output tensor name
  let conv1_1_in_name  = "140326425860192"
      fc6_out_name     = "140326200777584"
      softmax_out_name = "140326200803680"

  -- Load ONNX model data
  model_data <- makeModelDataFromONNX (optModelPath opt)

  -- Specify inputs and outputs
  vpt <- makeVariableProfileTable
           [(conv1_1_in_name, DTypeFloat, input_dims)]
           [fc6_out_name, softmax_out_name]
           model_data
  optimizeModelData model_data vpt

  -- Construct computation primitive list and memories
  model <- makeModel vpt model_data "mkldnn"

  -- Copy input image data to model's input array
  writeBuffer model conv1_1_in_name [image_data]

  -- Run inference
  run model

  -- Get output
  ([fc6_out] :: [V.Vector Float]) <- readBuffer model fc6_out_name
  putStr "fc6_out: "
  forM_ [0..4] $ \i -> do
    putStr $ show $ fc6_out V.! i
    putStr " "
  putStrLn "..."

  ([softmax_out] :: [V.Vector Float]) <- readBuffer model softmax_out_name

  categories <- liftM lines $ readFile (optSynsetWordsPath opt)
  let k = 5
  scores <- forM [0 .. V.length softmax_out - 1] $ \i -> do
    return (i, softmax_out V.! i)
  printf "top %d categories are:\n" k
  forM_ (take k $ sortBy (flip (comparing snd)) scores) $ \(i,p) -> do
    printf "%d %f %s\n" i p (categories !! i)

-- -------------------------------------------------------------------------

data Options
  = Options
  { optInputImagePath  :: FilePath
  , optModelPath       :: FilePath
  , optSynsetWordsPath :: FilePath
  }

optionsParser :: Parser Options
optionsParser = Options
  <$> inputImageOption
  <*> modelPathOption
  <*> synsetWordsPathOption
  where
    inputImageOption = strOption
      $  long "input-image"
      <> short 'i'
      <> metavar "PATH"
      <> help "input image path"
      <> value "data/Light_sussex_hen.jpg"
      <> showDefault
    modelPathOption = strOption
      $  long "model"
      <> short 'm'
      <> metavar "PATH"
      <> help "onnx model path"
      <> value "data/VGG16.onnx"
      <> showDefault
    synsetWordsPathOption = strOption
      $  long "synset-words"
      <> short 's'
      <> metavar "PATH"
      <> help "synset words path"
      <> value "data/synset_words.txt"
      <> showDefault

parserInfo :: ParserInfo Options
parserInfo = info (helper <*> versionOption <*> optionsParser)
  $  fullDesc
  <> header "vgg16_example - an example program of Menoh haskell binding"
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

-- Note that VGG16.onnx assumes BGR image
reorderToNCHW :: Picture.Image Picture.PixelRGB8 -> VS.Vector Float
reorderToNCHW img = VS.generate (3 * Picture.imageHeight img * Picture.imageWidth img) f
  where
    f i =
      case Picture.pixelAt img x y of
        Picture.PixelRGB8 r g b ->
          case ch of
            0 -> fromIntegral b
            1 -> fromIntegral g
            2 -> fromIntegral r
            _ -> undefined
      where
        (ch,m) = i `divMod` (Picture.imageWidth img * Picture.imageHeight img)
        (y,x) = m `divMod` Picture.imageWidth img

-- -------------------------------------------------------------------------
