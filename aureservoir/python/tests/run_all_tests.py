import unittest
import glob
all_tests = glob.glob( 'test_*.py' )

main_suite = unittest.TestSuite()

print
 
for test in all_tests:
  print 'Loading',test
  module = test.replace(".py","")
  exec 'import %s' % ( module )
  exec 'suite = unittest.TestLoader().loadTestsFromModule( %s )' % ( module );
  main_suite.addTest( suite )
  print

print

unittest.TextTestRunner(verbosity=2).run( main_suite ).wasSuccessful
