CC=gcc
CFLAGS=-Wall -Werror -g
LFLAGS=-L ./include/ -l m -l png 
OUTEXEC=bin/plpo
OBJECTS=src/main.c src/cli.c src/image.c src/interpolate.c src/plpo.c

.PHONY: build
build:
	${CC} ${CFLAGS} ${OBJECTS} -o ${OUTEXEC} ${LFLAGS}

run: build
	./bin/plpo 

gdb: build
	gdb --tui \
		--args \
		./bin/plpo 

valgrind: build
	valgrind --leak-check=full --track-origins=yes \
		./bin/plpo 

cubuild:
	nvcc src/main.cu src/image.c src/interpolate.cu src/plpo.cu \
		-g \
		-o bin/curun \
		-L ./include/ -l png

curun: cubuild
	@./bin/curun

cusanitizer: cubuild
	compute-sanitizer \
		./bin/curun

.PHONY: clean
clean:
	rm -r ./build/result.png ./bin/*
