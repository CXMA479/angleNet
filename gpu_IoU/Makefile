
CC=gcc-6

FLAG= -shared -pthread -fPIC -O3 -Wall -fno-strict-aliasing
INCLUDE = -I/usr/include/python2.7

src=cytool

target=$(src).so
all:$(target)

$(target):$(src).c
	$(CC)  $(FLAG) $(INCLUDE) -o $@ $^
