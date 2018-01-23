"""
     for infer_shape
       Chen Y. Liang
       Apr 28, 2017
"""


def mydic(a,d=0):
	print a
	print d



arg={'a':1,'d':-10}
mydic(**arg)

print 'now swap them'
arg={'d':-10,'a':1}
mydic(**arg)




