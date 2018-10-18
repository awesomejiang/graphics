#version 330 core
out vec4 fragColor;

in vec2 texCoord;

uniform sampler2D card;
uniform sampler2D glow;

void main()
{
    vec4 cardColor = texture(card, texCoord);
    vec4 glowColor = texture(glow, texCoord);

    fragColor = cardColor.a<0.1? glowColor: cardColor;
    //fragColor = glowColor;
    if(fragColor.a < 0.1)
        discard;
}