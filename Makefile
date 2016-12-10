CC=gcc
CFLAGS=-Wall -Werror -g
LFLAGS=-L ./include/ -l m -l png 
OUTEXEC=bin/plpo
OBJECTS=src/main.c src/cli.c src/image.c src/interpolate.c src/plpo.c
INPUT=build/original.png
TABLE=build/table.png
OUTPUT=build/output.png

.PHONY: build
build:
	@${CC} ${CFLAGS} ${OBJECTS} -o ${OUTEXEC} ${LFLAGS}

run: build
	@./bin/plpo ${INPUT} ${TABLE} ${OUTPUT}

gdb: build
	gdb --tui \
		--args \
		./bin/plpo 

valgrind: build
	valgrind --leak-check=full --track-origins=yes \
		./bin/plpo 

cubuild:
	@nvcc src/main.cu src/image.c src/interpolate.cu src/plpo.cu \
		-g \
		-o bin/curun \
		-L ./include/ -l png

curun: cubuild
	@./bin/curun ${INPUT} ${TABLE} ${OUTPUT}

cusanitizer: cubuild
	compute-sanitizer \
		./bin/curun

.PHONY: clean
clean:
	rm ./build/output.png ./bin/*
