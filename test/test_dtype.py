import unittest
from shrimpgrad.dtype import type_promotion 
from shrimpgrad.dtype import (
  b1_, i_, f_, c_,
  u8_, u16_, u32_, u64_,
  i8_, i16_, i32_, i64_,
  f16_, bf16_, f32_, f64_,
  c64_, c128_
)

class TestDtype(unittest.TestCase):
  def test_bool_int_promotion(self):
    # bool promotes to int since int can represent all bool values
    self.assertEqual(type_promotion(b1_, i_), i_)
    self.assertEqual(type_promotion(i_, b1_), i_)

  def test_int_to_fixed_width_promotion(self):
    # Python int promotes to fixed width integers (int8/uint8)
    # since they can represent a subset of int values
    self.assertEqual(type_promotion(i_, i8_), i8_)
    self.assertEqual(type_promotion(i8_, i_), i8_)
    self.assertEqual(type_promotion(i_, u8_), u8_)
    self.assertEqual(type_promotion(u8_, i_), u8_)

  def test_8bit_to_16bit_promotion(self):
    # 8-bit integers promote to 16-bit integers since they can represent
    # all 8-bit values plus additional range
    self.assertEqual(type_promotion(i8_, i16_), i16_)
    self.assertEqual(type_promotion(i16_, i8_), i16_)
    self.assertEqual(type_promotion(u8_, u16_), u16_)
    self.assertEqual(type_promotion(u16_, u8_), u16_)
    # uint8 can promote to int16 since int16 can represent all uint8 values
    self.assertEqual(type_promotion(u8_, i16_), i16_)
    self.assertEqual(type_promotion(i16_, u8_), i16_)

  def test_16bit_to_32bit_promotion(self):
    # 16-bit integers promote to 32-bit integers for the same reason as above
    self.assertEqual(type_promotion(i16_, i32_), i32_)
    self.assertEqual(type_promotion(i32_, i16_), i32_)
    self.assertEqual(type_promotion(u16_, u32_), u32_)
    self.assertEqual(type_promotion(u32_, u16_), u32_)
    self.assertEqual(type_promotion(u16_, i32_), i32_)
    self.assertEqual(type_promotion(i32_, u16_), i32_)

  def test_32bit_to_64bit_promotion(self):
    # 32-bit integers promote to 64-bit integers for the same reason as above
    self.assertEqual(type_promotion(i32_, i64_), i64_)
    self.assertEqual(type_promotion(i64_, i32_), i64_)
    self.assertEqual(type_promotion(u32_, u64_), u64_)
    self.assertEqual(type_promotion(u64_, u32_), u64_)
    self.assertEqual(type_promotion(u32_, i64_), i64_)
    self.assertEqual(type_promotion(i64_, u32_), i64_)

  def test_64bit_to_float_promotion(self):
    # 64-bit integers promote to float since float can approximate integer values
    # Note: Some precision may be lost for very large integers
    self.assertEqual(type_promotion(i64_, f_), f_)
    self.assertEqual(type_promotion(f_, i64_), f_)
    self.assertEqual(type_promotion(u64_, f_), f_)
    self.assertEqual(type_promotion(f_, u64_), f_)

  def test_float_promotions(self):
    # Float promotes to complex since complex can represent all real numbers
    # Float also promotes to float16/bfloat16 for more specific precision
    self.assertEqual(type_promotion(f_, c_), c_)
    self.assertEqual(type_promotion(c_, f_), c_)
    self.assertEqual(type_promotion(f_, f16_), f16_)
    self.assertEqual(type_promotion(f16_, f_), f16_)
    self.assertEqual(type_promotion(f_, bf16_), bf16_)
    self.assertEqual(type_promotion(bf16_, f_), bf16_)

  def test_complex_promotions(self):
    # Complex promotes to complex64 for fixed precision
    self.assertEqual(type_promotion(c_, c64_), c64_)
    self.assertEqual(type_promotion(c64_, c_), c64_)

  def test_float_precision_promotions(self):
    # Lower precision floats promote to higher precision floats
    # float32 can also promote to complex64 since complex can represent all reals
    self.assertEqual(type_promotion(f16_, f32_), f32_)
    self.assertEqual(type_promotion(f32_, f16_), f32_)
    self.assertEqual(type_promotion(bf16_, f32_), f32_)
    self.assertEqual(type_promotion(f32_, bf16_), f32_)
    self.assertEqual(type_promotion(f32_, f64_), f64_)
    self.assertEqual(type_promotion(f64_, f32_), f64_)
    self.assertEqual(type_promotion(f32_, c64_), c64_)
    self.assertEqual(type_promotion(c64_, f32_), c64_)

  def test_complex_precision_promotions(self):
    # Lower precision complex promotes to higher precision complex
    # float64 promotes to complex128 since complex can represent all reals
    self.assertEqual(type_promotion(f64_, c128_), c128_)
    self.assertEqual(type_promotion(c128_, f64_), c128_)
    self.assertEqual(type_promotion(c64_, c128_), c128_)
    self.assertEqual(type_promotion(c128_, c64_), c128_)

  def test_promotion_properties(self):
    # Test associativity property: (a->b)->c == a->(b->c)
    self.assertEqual(type_promotion(type_promotion(i8_, i16_), i32_),
                    type_promotion(i8_, type_promotion(i16_, i32_)))

    # Same type promotion should return the same type
    self.assertEqual(type_promotion(f32_, f32_), f32_)
    self.assertEqual(type_promotion(i32_, i32_), i32_)
    self.assertEqual(type_promotion(c64_, c64_), c64_)
  
