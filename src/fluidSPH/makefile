CXX = nvcc
CU_FLAGS = -std=c++14 -x cu -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -lineinfo
CPP_FLAGS = -std=c++14 -lineinfo
HEADERS = /home/jiawei/Desktop/graphics/src/headers

PROGS = main

FS = fluidSystem gridcells particle
UTIL = utility vec_float
GL = framebuffer mesh texture shader window camera#scene

FS_PATH = $(HEADERS)/particle
UTIL_PATH = $(HEADERS)/utility
GL_PATH = $(HEADERS)/myGL

LINKS = -lcuda -lcudart -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXinerama -lXxf86vm -lXcursor -lXi -ldl -lassimp


INCLUDES = $(foreach path, $(FS_PATH) $(UTIL_PATH) $(GL_PATH), -I $(path)/include)
PROGS_OBJ = $(patsubst %, obj/%.o, $(PROGS))
FS_OBJ = $(patsubst %, obj/%.o, $(FS))
UTIL_OBJ = $(patsubst %, obj/%.o, $(UTIL))
GL_OBJ = $(patsubst %, obj/%.o, $(GL))
OBJS = $(PROGS_OBJ) $(FS_OBJ) $(UTIL_OBJ) $(GL_OBJ)
DIRS = obj

define printMeg
	@tput setaf 3
	@echo $1
	@tput sgr0
endef

all: $(DIRS) $(PROGS)

$(PROGS): %: %.cpp $(FS_OBJ) $(UTIL_OBJ) $(GL_OBJ)
	$(call printMeg, "Building main program: $(PROGS) ...")

	$(CXX) $(CU_FLAGS) -dc $(INCLUDES) -o obj/$@.o $<
	$(CXX) $(CPP_FLAGS) $(INCLUDES) $(OBJS) $(GL_PATH)/src/glad.c -o $@ $(LINKS)

	$(call printMeg, "Built!\n")

$(FS_OBJ): obj/%.o: $(FS_PATH)/src/%.cpp $(FS_PATH)/include/%.h
	$(call printMeg, "Building particle: obj/$*.o ...")

	$(CXX) $(CU_FLAGS) $(INCLUDES) -dc -o $@ $<

	$(call printMeg, "Built!\n")

$(UTIL_OBJ): obj/%.o: $(UTIL_PATH)/src/%.cpp $(UTIL_PATH)/include/%.h
	$(call printMeg, "Building utility: obj/$*.o ...")

	$(CXX) $(CU_FLAGS) $(INCLUDES) -dc -o $@ $<

	$(call printMeg, "Built!\n")

$(GL_OBJ): obj/%.o: $(GL_PATH)/src/%.cpp $(GL_PATH)/include/%.h
	$(call printMeg, "Building myGL: obj/$*.o ...")

	$(CXX) $(CPP_FLAGS) $(INCLUDES) -c -o $@ $<

	$(call printMeg, "Built!\n")


$(DIRS):
	mkdir -p $@


.PHONY: clean
clean:
	rm -rf $(PROGS) $(DIRS)