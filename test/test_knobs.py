import unittest

import os

class TestKnobs(unittest.TestCase):
  def test_knob_with_env_set(self):
    os.environ['DEBUG'] = '4'
    from shrimpgrad.knobs import DEBUG
    assert DEBUG == 4, "not right"

  def test_knob_default(self):
    from shrimpgrad.knobs import DEBUG
    assert DEBUG == 0, "not right"
  
  def test_knob_with_context(self):
    from shrimpgrad.knobs import Knobs, DEBUG
    assert DEBUG == 0, 'pre context not correct default'
    with Knobs(DEBUG=4):
      assert DEBUG == 4, 'context not altered'
    assert DEBUG == 0, 'context change persisted'