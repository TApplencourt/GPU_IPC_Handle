CXX?=mpicxx
CXXFLAGS?=-fiopenmp -fopenmp-targets=spir64

FILES=device_buffer.cpp device_ptr.cpp shared_buffer.cpp shared_ptr.cpp

RUN_TARGET=$(FILES:%.cpp=run_%)
EXE_FILES=$(FILES:%.cpp=%.exe)

run: $(RUN_TARGET)
exe: $(EXE_FILES)

.PHONY: clean run $(EXE_FILES)

clean: 
	rm -f $(EXE_FILES)

$(EXE_FILES): %.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(RUN_TARGET): run_%: %.exe
	-mpirun -n 2 ./$<
