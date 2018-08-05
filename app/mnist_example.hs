{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module Main (main) where

import qualified Codec.Picture as Picture
import qualified Codec.Picture.Types as Picture
import Control.Applicative
import Control.Monad
import Data.Monoid
import qualified Data.Vector.Generic as VG
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

  images <- forM [(0::Int)..9] $ \i -> do
    let fname :: String
        fname = printf "%d.png" i
    ret <- Picture.readImage $ input_dir </> fname
    case ret of
      Left e -> error e
      Right img -> return (Picture.extractLumaPlane $ Picture.convertRGB8 img, i, fname)

  let batch_size  = length images
      channel_num = 1
      height = 28
      width  = 28
      category_num = 10

      input_dims, output_dims :: Dims
      input_dims  = [batch_size, channel_num, height, width]
      output_dims = [batch_size, category_num]

      -- Aliases to onnx's node input and output tensor name
      mnist_in_name  = "139900320569040"
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
  writeBuffer model mnist_in_name [VG.map fromIntegral (Picture.imageData img) :: VS.Vector Float | (img,_,_) <- images]

  -- Run inference
  run model

  -- Get output
  (vs :: [VS.Vector Float]) <- readBuffer model mnist_out_name

  -- Examine the results
  forM_ (zip images vs) $ \((_img,expected,fname), scores) -> do
    let guessed = VG.maxIndex scores
    putStrLn fname
    printf "Expected: %d Guessed: %d\n" expected guessed
    putStrLn $ "Scores: " ++ show (zip [(0::Int)..] (VG.toList scores))
    putStrLn $ "Probabilities: " ++ show (zip [(0::Int)..] (VG.toList (softmax scores)))
    putStrLn ""

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

softmax :: (Real a, Floating a, VG.Vector v a) => v a -> v a
softmax v | VG.null v = VG.empty
softmax v = VG.map (/ s) v'
  where
    m = VG.maximum v
    v' = VG.map (\x -> exp (x - m)) v
    s = VG.sum v'

-- -------------------------------------------------------------------------
