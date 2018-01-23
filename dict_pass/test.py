

def test(a,b,c):
  print('a:%d'%a)
  print('b:%d'%b)
  print('c:%d'%c)


if __name__ == '__main__':
  d={'a':1,'c':-10,'b':100}
  test(**d)
