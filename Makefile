CC=gcc
CFLAGS=-Wall -Werror
LFLAGS=-L ./include/ -l png
OUTEXEC=bin/plpo
OBJECTS=src/main.c src/cli.c src/image.c src/interpolate.c src/plpo.c
#OBJECTS=src/*.c

.PHONY: build
build:
	${CC} ${CFLAGS} ${OBJECTS} -o ${OUTEXEC} ${LFLAGS}

run: build
	./bin/plpo

.PHONY: clean
clean:
	rm -r ./build/* ./bin/*

