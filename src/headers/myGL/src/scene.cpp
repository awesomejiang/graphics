#include "scene.h"

Scene::Scene(int width, int height)
: gbuffer{width, height, 
    std::make_pair("gPosition", GL_RGB16F),
    std::make_pair("gNormal", GL_RGB16F),
    std::make_pair("gSpec", GL_RGBA16F)},
  gbufferBlend{width, height.
    std::make_pair("gDiff", GL_RGBA16F)},
  Shader{"shader/object.vs", "shader/object.fs"},
  skyboxShader{"shader/skybox.vs", "shader/skybox.fs"} {
  	
  }

void Scene::loadModel(std::string const &dir){
	std::cout << "Loading " + dir + " ..." << std::endl;

	auto obj = dir + "/" + dir.substr(dir.find_last_of("/")+1) + ".obj";
	models.emplace_back(obj);
}


void Scene::loadMesh(std::string const &dir){
	std::cout << "Loading " + dir + " ..." << std::endl;

	auto vertices = readVertices(dir+"/vertices.txt");
	auto indices = readIndices(dir+"/indices.txt");
	auto textures = readTextures(dir+"/textures.txt");

	meshs.emplace_back(vertices, indices, textures);
}

void Scene::loadSkybox(std::string const &dir){
	std::cout << "Loading " + dir + " ..." << std::endl;

	auto vertices = readVertices(dir+"/vertices.txt");
	auto indices = readIndices(dir+"/indices.txt");
	auto textures = readTextures(dir+"/textures.txt");

	skybox = Mesh{vertices, indices, textures};
}

void Scene::loadLight(std::string const &dir){
	std::cout << "Loading " + dir + " ..." << std::endl;

	//directional lights
	std::ifstream ifs1{dir+"/directional.txt"};
	if(ifs.is_open()){
		std::string line;
		while(getline(ifs1, line)){
			glm::vec3 vec;
			std::stringstream ss(line);
			ss >> vec.x;
			ss >> vec.y;
			ss >> vec.z;
			directionals.push_back(vec);
		}
	}

	//point lights
	std::ifstream ifs2{dir+"/point.txt"};
	if(ifs.is_open()){
		std::string line;
		while(getline(ifs2, line)){
			glm::vec3 vec;
			std::stringstream ss(line);
			ss >> vec.x;
			ss >> vec.y;
			ss >> vec.z;
			points.push_back(vec);
		}
	}

	//spot lights
	std::ifstream ifs3{dir+"/spot.txt"};
	if(ifs.is_open()){
		std::string line;
		while(getline(ifs3, line)){
			glm::vec3 vec;
			std::stringstream ss(line);
			ss >> vec.x;
			ss >> vec.y;
			ss >> vec.z;
			spots.push_back(vec);
		} 
	}
}

/* private helper functions */
std::vector<Vertex> Scene::readVertices(std::string const &file){
	//read vertices
	std::ifstream ver{file};
	if(!ver.is_open()){
		throw std::runtime_error("error: cannot open vertices file.");
	}
	std::string line;
	std::vector<Vertex> vertices;
	while(getline(ver, line)){
		std::stringstream ss(line);
		Vertex v;
		ss >> v.position[0];
		ss >> v.position[1];
		ss >> v.position[2];
		ss >> v.normal[0];
		ss >> v.normal[1];
		ss >> v.normal[2];
		ss >> v.texCoords[0];
		ss >> v.texCoords[1];
		vertices.push_back(v);
	}

	return vertices;
}

std::vector<unsigned int> Scene::readIndices(std::string const &file){
	//try read indices
	std::ifstream ind{file};
	std::vector<unsigned int> indices;
	if(ind.is_open()){
		std::string line;
		while(getline(ind, line)){
			std::stringstream ss(line);
			unsigned int val;
			indices.push_back(val);
		}
	}

	return indices;
}

std::vector<Texture> Scene::readTextures(std::string const &file){
	//try read textures
	std::ifstream tex{file};
	std::vector<Texture> textures;
	if(tex.is_open()){
		std::string line;
		while(getline(tex, line)){
			//read type and name;
			std::stringstream ss(line);
			std::string type, name;
			ss >> type;
			if(!(ss >> name)){
				name == "";
			}
			
			ss.str(std::string());
			ss.clear();
			getline(tex, line);
			ss << line;
			std::string str;
			std::vector<std::string> paths;
			while(ss >> str){
				paths.push_back(dir+"/"+str);
			}
			if(paths.size() == 1)
				textures.emplace_back(paths[0], type, name);
			else
				textures.emplace_back(paths, type, name);
		}
	}

	return textures;
}