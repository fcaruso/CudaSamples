#include <iostream>
#include <vector>

#include <gl/freeglut.h>

#include "PhysicsEngine.h"

const int	WINDOW_WIDTH	= 480;
const int	WINDOW_HEIGHT	= 480;
const float BOX_HALF_SIZE	= 5.f;
const float BOX_THICKNESS   = 0.5f;


const unsigned NUM_OF_SPEHERES = 10;


void cbReshape(int w, int h)
{
	glViewport(0,0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float ar=(float) w/h;//aspect ratio
	// increase viewport size just a bit to draw walls
	
	float size = BOX_HALF_SIZE + BOX_THICKNESS;
	glOrtho( -size, size, -BOX_THICKNESS, (-BOX_THICKNESS + 2*size)/ar, -5., 5. );
	glMatrixMode(GL_MODELVIEW);
}

void drawWalls()
{
	glPushAttrib(GL_CURRENT_BIT);
	glColor3f(0.7, 0.7, 0.7);

	// left
	glRectf(-BOX_HALF_SIZE - BOX_THICKNESS, -BOX_THICKNESS,
		-BOX_HALF_SIZE, 2*BOX_HALF_SIZE);
	//down
	glRectf( -BOX_HALF_SIZE - BOX_THICKNESS, -BOX_HALF_SIZE,
			BOX_HALF_SIZE + BOX_THICKNESS, 0);
	// right
	glRectf( BOX_HALF_SIZE, -BOX_THICKNESS,
		BOX_HALF_SIZE + BOX_THICKNESS, 2*BOX_HALF_SIZE );

	glPopAttrib();
}

void cbDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	drawWalls();


	////draw spheres
	//for (int i=0; i<g_sphere.size(); i++)
	//	g_sphere[i].draw();

	glutSwapBuffers();
}

void cbKey(unsigned char c, int x, int y)
{
	switch(c)
	{
	case 'q': case 'Q':case 27:
		glutLeaveMainLoop();
		break;
	}
}

void initPhysics()
{
	//for (int i=0; i<NUM_OF_SPEHERES; i++)
	{
		Sphere s1(vec2(-2.f, 4.f), 0.5f);
		s1.m_color = vec3(0.5, 0.2, 0.3);
		g_sphere.push_back(s1);

		Sphere s2(3.f, 2.f, 0.7f);
		g_sphere.push_back(s2);

	}
}
int main(int argc, char** argv)
{
	initPhysics();
	glutInit(&argc,argv);
	glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
	glutCreateWindow("Spheres2D");
	glutReshapeFunc(cbReshape);
	glutDisplayFunc(cbDisplay);
	glutKeyboardFunc(cbKey);
	glutMainLoop();
	return 0;
}