CXX = nvcc
CU_FLAGS = -std=c++14 -x cu -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -lineinfo
CPP_FLAGS = -std=c++14
HEADERS = /home/jiawei/Dropbox/github/hsCrawler/src/headers

PROGS = main

PS = click tailing square gravity
UTIL = utility vec_float
FLUID = fluid
GL = framebuffer mesh texture shader scene

PS_PATH = $(HEADERS)/cudaParticle
UTIL_PATH = $(HEADERS)/utility
FLUID_PATH = $(HEADERS)/cudaFluid
GL_PATH = $(HEADERS)/myGL

LINKS = -lcuda -lcudart -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXinerama -lXxf86vm -lXcursor -lXi -ldl -lassimp


INCLUDES = $(foreach path, $(PS_PATH) $(UTIL_PATH) $(FLUID_PATH) $(GL_PATH), -I $(path)/include)
PROGS_OBJ = $(patsubst %, obj/%.o, $(PROGS))
PS_OBJ = $(patsubst %, obj/%.o, $(PS))
FLUID_OBJ = $(patsubst %, obj/%.o, $(FLUID))
UTIL_OBJ = $(patsubst %, obj/%.o, $(UTIL))
GL_OBJ = $(patsubst %, obj/%.o, $(GL))
OBJS = $(PROGS_OBJ) $(PS_OBJ) $(UTIL_OBJ) $(FLUID_OBJ) $(GL_OBJ)
DIRS = obj

define printMeg
	@tput setaf 3
	@echo $1
	@tput sgr0
endef

all: $(DIRS) $(PROGS)

$(PROGS): %: %.cpp $(PS_OBJ) $(UTIL_OBJ) $(FLUID_OBJ) $(GL_OBJ)
	$(call printMeg, "Building main program: $(PROGS) ...")

	$(CXX) $(CU_FLAGS) -dc $(INCLUDES) -o obj/$@.o $<
	$(CXX) $(CPP_FLAGS) $(INCLUDES) $(OBJS) $(GL_PATH)/src/glad.c -o $@ $(LINKS)

	$(call printMeg, "Built!\n")

$(PS_OBJ): obj/%.o: $(PS_PATH)/src/%.cpp $(PS_PATH)/include/%.h
	$(call printMeg, "Building cudaParticle: obj/$*.o ...")

	$(CXX) $(CU_FLAGS) $(INCLUDES) -dc -o $@ $<

	$(call printMeg, "Built!\n")

$(FLUID_OBJ): obj/%.o: $(FLUID_PATH)/src/%.cpp $(FLUID_PATH)/include/%.h
	$(call printMeg, "Building cudaFluid: obj/$*.o ...")

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