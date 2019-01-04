#include "gbuffer.h"

void Gbuffer::initTextures(std::vector<std::pair<std::string, unsigned int>> const &pairs){
	//init textures
	glBindFramebuffer(GL_FRAMEBUFFER, buffer);
	auto attachPoint = GL_COLOR_ATTACHMENT0;
	for(auto pair: pairs){
		auto name = pair.first;
		auto type = pair.second;

		//build texture
		unsigned int tex;
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		textures.push_back({tex, "texture2D", name});

		if(type == GL_RGB16F)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
		else if(type == GL_RGBA16F)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
		else if(type == GL_RGB)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
		else if(type == GL_RGBA)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture2D(GL_FRAMEBUFFER, attachPoint++, GL_TEXTURE_2D, tex, 0);
	}

	//Bind drawbuffers
	std::vector<unsigned int> attachments(pairs.size());
	std::iota(attachments.begin(), attachments.end(), GL_COLOR_ATTACHMENT0);
	glDrawBuffers(attachments.size(), attachments.data());

	//generate renderbuffer for depth and stencil test
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
		throw std::runtime_error("Error in Framebuffer::Framebuffer(): Framebuffer is not complete.");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void Gbuffer::bind() const{
	glBindFramebuffer(GL_FRAMEBUFFER, buffer);
}

void Gbuffer::unbind() const{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Gbuffer::setTextures(Mesh &mesh) const{
	for(auto texture: textures)
		mesh.setTexture(texture);
}

void Gbuffer::blitData(unsigned int target, unsigned int dataBit) const{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, buffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, dataBit, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void Gbuffer::blitData(Gbuffer const &gbuffer, unsigned int dataBit) const{
	blitData(gbuffer.getFramebuffer(), dataBit);
}