CXX = nvcc
FLAGS = -std=c++11 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -lineinfo

PROGS = main

CUS = utility spirit particle vec
CPPS = pack mesh texture shader scene card

CU_OBJS = $(patsubst %, obj/%.o, $(CUS))

CPP_OBJS = $(patsubst %, obj/%.o, $(CPPS))

LINKS = -lcuda -lcudart -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXinerama -lXxf86vm -lXcursor -lXi -ldl -lassimp

DIRS = obj

all: $(DIRS) $(PROGS)

$(PROGS): %: %.cpp $(CU_OBJS) $(CPP_OBJS)
	$(CXX) $(FLAGS) -dlink $(CU_OBJS) $(CPP_OBJS) -o obj/$@.o
	$(CXX) $(FLAGS) -Iinclude $@.cpp obj/$@.o $(CU_OBJS) $(CPP_OBJS) src/glad.c -o $@ $(LINKS)

$(CU_OBJS): obj/%.o: src/%.cu include/%.cuh
	$(CXX) $(FLAGS) -dc -Iinclude -o $@ $<

$(CPP_OBJS): obj/%.o: src/%.cpp include/%.h
	$(CXX) $(FLAGS) -c -Iinclude -o $@ $<

$(DIRS):
	mkdir -p $@

clean:
	rm -rf $(PROGS) $(DIRS)