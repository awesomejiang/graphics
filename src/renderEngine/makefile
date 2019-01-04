CXX = g++
CPP_FLAGS = -std=c++14
HEADERS = /home/jiawei/Desktop/graphics/src/headers

PROGS = main

GL = mesh model texture shader scene camera loader

GL_PATH = $(HEADERS)/myGL

LINKS = -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXinerama -lXxf86vm -lXcursor -lXi -ldl -lassimp


INCLUDES = $(foreach path, $(GL_PATH), -I $(path)/include)
PROGS_OBJ = $(patsubst %, obj/%.o, $(PROGS))
GL_OBJ = $(patsubst %, obj/%.o, $(GL))
OBJS = $(PROGS_OBJ) $(GL_OBJ)
DIRS = obj

define printMeg
	@tput setaf 3
	@echo $1
	@tput sgr0
endef

all: $(DIRS) $(PROGS)

$(PROGS): %: %.cpp $(GL_OBJ)
	$(call printMeg, "Building main program: $(PROGS) ...")

	$(CXX) $(CPP_FLAGS) -c $(INCLUDES) -o obj/$@.o $<
	$(CXX) $(CPP_FLAGS) $(INCLUDES) $(OBJS) $(GL_PATH)/src/glad.c -o $@ $(LINKS)

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