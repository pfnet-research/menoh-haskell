#!/usr/bin/env stack 
-- stack runghc --package conduit --package conduit-extra --package http-conduit
{-# OPTIONS_GHC -Wall #-}
import Control.Monad.Trans.Resource
import Data.Conduit.Binary (sinkFile)
import Network.HTTP.Simple

main :: IO ()
main = do
  let downloadTo :: String -> FilePath -> IO ()
      downloadTo req fname = do
        putStrLn $ req ++ " -> " ++ fname
        request <- parseRequest req
        runResourceT $ httpSink request $ \_ -> sinkFile fname
  downloadTo "https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1" "./data/VGG16.onnx"
  downloadTo "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt" "./data/synset_words.txt"
  downloadTo "https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg" "./data/Light_sussex_hen.jpg"
