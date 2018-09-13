CXX = nvcc
CU_FLAGS = -std=c++14 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -lineinfo
CPP_FLAGS = -std=c++14
HEADERS = /home/jiawei/Dropbox/github/hsCrawler/src/headers

PROGS = main

CUS = square gravity flame
CUS += utility vec_float
CPPS = framebuffer mesh texture shader scene

CU_OBJS = $(patsubst %, obj/%.o, $(CUS))
CU_INCLUDES = $(HEADERS)/cudaParticle/include
CU_SRCS = $(HEADERS)/cudaParticle/src
CPP_OBJS = $(patsubst %, obj/%.o, $(CPPS))
CPP_INCLUDES = $(HEADERS)/myGL/include
CPP_SRCS = $(HEADERS)/myGL/src

LINKS = -lcuda -lcudart -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXinerama -lXxf86vm -lXcursor -lXi -ldl -lassimp

DIRS = obj


all: $(DIRS) $(PROGS)

define printMeg
	@tput setaf 3
	@echo $1
	@tput sgr0
endef

$(PROGS): %: %.cpp $(CU_OBJS) $(CPP_OBJS)
	$(call printMeg, "Building $(PROGS) ...")

	$(CXX) $(CU_FLAGS) -x cu -dc -I $(CU_INCLUDES) -I $(CPP_INCLUDES) -o obj/$@.o $<
	$(CXX) $(CPP_FLAGS) -I $(CU_INCLUDES) -I $(CPP_INCLUDES) \
		$(CU_OBJS) $(CPP_OBJS) obj/$@.o $(CPP_SRCS)/glad.c -o $@ $(LINKS)

	$(call printMeg, "Built!\n")

$(CU_OBJS): obj/%.o: $(CU_SRCS)/%.cpp $(CU_INCLUDES)/%.h
	$(call printMeg, "Building obj/$*.o ...")

	$(CXX) $(CU_FLAGS) -x cu -dc -I $(CU_INCLUDES) -I $(CPP_INCLUDES) -o $@ $<

	$(call printMeg, "Built!\n")

$(CPP_OBJS): obj/%.o: $(CPP_SRCS)/%.cpp $(CPP_INCLUDES)/%.h
	$(call printMeg, "Building obj/$*.o ...")

	$(CXX) $(CPP_FLAGS) -c -I $(CU_INCLUDES) -I $(CPP_INCLUDES) -o $@ $<

	$(call printMeg, "Built!\n")


$(DIRS):
	mkdir -p $@

clean:
	rm -rf $(PROGS) $(DIRS)


	#$(CXX) $(FLAGS) -dlink $(CU_OBJS) $(CPP_OBJS) -o obj/$@_links.o