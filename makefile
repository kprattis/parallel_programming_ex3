CC = nvcc
CXX = g++

cpp_files = $(wildcard src/*.cpp lib/*.cpp)
cu_files = $(wildcard src/*.cu lib/*.cu)

FLAGS = -O2 -Ilib -g

OUT = build/fglt_cuda.out

OUT_SEQ = build/fglt_seq.out

all: $(OUT)

SEQ: $(OUT_SEQ)

DEBUG: $(cpp_files) $(cu_files)
	mkdir -p build
	$(CC) $^ $(FLAGS) -DDEBUG -o $(OUT)

$(OUT_SEQ): $(cpp_files)
	mkdir -p build
	$(CXX) $^ $(FLAGS) -DSEQ -o $@


$(OUT): $(cpp_files) $(cu_files)
	mkdir -p build
	$(CC) $^ $(FLAGS) -o $@

test: $(OUT) $(OUT_SEQ)
	echo "Testing GPU version:"
	build/fglt_cuda.out assets/dictionary28.mtx

	echo "Testing CPU sequential version"
	build/fglt_seq.out assets/dictionary28.mtx

clean:
	rm build/*.out