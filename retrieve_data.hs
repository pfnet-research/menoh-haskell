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
  downloadTo "https://preferredjp.box.com/shared/static/o2xip23e3f0knwc5ve78oderuglkf2wt.onnx" "./data/vgg16.onnx"
  downloadTo "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt" "./data/synset_words.txt"
  downloadTo "https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg" "./data/Light_sussex_hen.jpg"
