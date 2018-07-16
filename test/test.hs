{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}

import qualified Data.Vector as V
import qualified Data.Vector.Generic as VG
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Unboxed as VU
import Foreign

import Test.Tasty.HUnit
import Test.Tasty.TH

import Menoh

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
-- Test harness

main :: IO ()
main = $(defaultMainGenerator)
