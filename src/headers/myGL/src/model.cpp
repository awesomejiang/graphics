#include "model.h"

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Model::Model(std::string path)
: directory(path.substr(0, path.find_last_of('/'))){
	Assimp::Importer importer;
	scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
	if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		throw std::runtime_error(std::string("ERROR::ASSIMP::") + std::string(importer.GetErrorString()));
	
	processNode(scene->mRootNode);
}


void Model::processNode(aiNode *node){
	//push all meshes
	for(auto i=0; i<node->mNumMeshes; ++i){
		auto mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh));
	}

	//process all its children
	for(auto i=0; i<node->mNumChildren; ++i)
		processNode(node->mChildren[i]);
}


Mesh Model::processMesh(aiMesh *mesh){
	//process vertices
	std::vector<Vertex> vertices;
	for(auto i=0; i<mesh->mNumVertices; ++i){
		Vertex vertex;
		vertex.position = {
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		};
		vertex.normal = {
			mesh->mNormals[i].x,
			mesh->mNormals[i].y,
			mesh->mNormals[i].z
		};
		if(mesh->mTextureCoords[0])
			vertex.texCoords = {
				mesh->mTextureCoords[0][i].x,
				mesh->mTextureCoords[0][i].y
			};
		else
			vertex.texCoords = {0.0f, 0.0f};

		vertices.push_back(vertex);
	}

	//process indices
	std::vector<unsigned int> indices;
	for(auto i=0; i<mesh->mNumFaces; ++i){
		aiFace face = mesh->mFaces[i];
		for(auto j=0; j<face.mNumIndices; ++j)
			indices.push_back(face.mIndices[j]);
	}

	//process textures
	std::vector<Texture> textures;
	if(mesh->mMaterialIndex >= 0){
		aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
		//diffuse
		std::vector<Texture> diffuseMaps = loadMaterialTextures(mat, aiTextureType_DIFFUSE, "textureDiffuse");
		textures.assign(diffuseMaps.begin(), diffuseMaps.end());
		//specular
		std::vector<Texture> specularMaps = loadMaterialTextures(mat, aiTextureType_SPECULAR, "textureSpecular");
		textures.insert(textures.begin(), specularMaps.begin(), specularMaps.end());
		//normal
		std::vector<Texture> normalMaps = loadMaterialTextures(mat, aiTextureType_NORMALS, "textureNormal");
		textures.insert(textures.begin(), normalMaps.begin(), normalMaps.end());
		//height
		std::vector<Texture> heightMaps = loadMaterialTextures(mat, aiTextureType_HEIGHT, "textureHeight");
		textures.insert(textures.begin(), heightMaps.begin(), heightMaps.end());
	}

	return Mesh(vertices, indices, textures);
}


std::vector<Texture> Model::loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName){
	std::vector<Texture> textures;
	for(auto i=0; i<mat->GetTextureCount(type); ++i){
		//get texture path
		aiString aStr;
		mat->GetTexture(type, i, &aStr);

		//check if path is already loaded
		std::string str(aStr.C_Str());
		if(loadedTexture.find(str) == loadedTexture.end())
			loadedTexture.emplace(str, Texture(directory+"/"+str, typeName));

		//set texture into textures from map
		textures.push_back(loadedTexture[str]);
	}

	return textures;
}

void Model::setTexture(Texture const &texture){
	for(auto &mesh: meshes)
		mesh.setTexture(texture);
}

void Model::draw(Shader const &shader) const{
	for(auto &mesh: meshes)
		mesh.draw(shader);
}