#ifndef PHYSICS_ENGINE
#define PHYSICS_ENGINE

#include <GL/freeglut.h>
#include <glm/glm.hpp>


#include <vector>
#define _USE_MATH_DEFINES 
#include <math.h>

using glm::vec2;
using glm::vec3;


struct Sphere
{
	float	m_radius;

	vec2	m_x;		// position
	vec2	m_v;		// velocity
	vec2	m_a;		// acceleration
	vec2	m_xprev;	// previous position
	float	m_im;		// 1/mass
	vec3	m_color;

	Sphere()
	{
		m_radius = 1.f;
		m_x = vec2(0.f,0.f);
		m_v = vec2(0.f, 0.f);
		m_a = vec2(0.f, 0.f);
		m_im = 1.f;
		m_color=vec3(1.f,1.f,1.f);
	}

	Sphere(float x, float y, float radius)
	{
		m_radius = radius;
		m_x.x = x;
		m_x.y = y;

		m_v = vec2(0.,0.);
		m_a = vec2(0.f, 0.f);
		m_im = 1.f/(M_PI*radius*radius);
		m_color=vec3(1.f,1.f,1.f);
	}
	Sphere(vec2 position, float radius)
	{
		m_x=position;
		m_a=m_v=vec2(0.f, 0.f);
		m_radius=radius;
		m_im = 1.f/(M_PI*radius*radius);
		m_color=vec3(1.f,1.f,1.f);
	}

	void draw()
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glTranslatef( m_x[0], m_x[1], 0 );
		glPushAttrib(GL_ENABLE_BIT);
		glColor3fv(&m_color[0]);
		glutSolidSphere(m_radius, 16, 16);
		glPopAttrib();
		glPopMatrix();
	}
};

struct Box
{
	vec2	m_min;
	vec2	m_max;
	float	m_thickness;

	Box( float horizontal_size, float vertical_size, float thickness=0.1 )
	{
		m_min.x = -horizontal_size /2.f;
		m_min.y = 0.f;

		m_max.x = horizontal_size/2.f;
		m_max.y = vertical_size;
	}
	
	void draw(bool drawInWorldCoordinate=true)
	{
		if (drawInWorldCoordinate)
		{
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
		}
	
		// left
		glRectf( m_min.x - m_thickness, -m_thickness,
			m_min.x, m_max.y);

		//down
		glRectf( m_min.x - m_thickness, -m_thickness,
				m_max.x + m_thickness, 0);
		// right
		glRectf( m_max.x, -m_thickness,
			 m_max.x + m_thickness, m_max.y );

		if (drawInWorldCoordinate)
		{
			glPopMatrix();
		}
	}
};

struct PhysicsEngine
{
private:
	std::vector<Sphere>	m_sphere;
	Box					m_box;

	void				applyGravity(float g= (-1.0f) );
	void				computeForces();
	void				integrate(float dt);
	void				manageCollisions(float dt);

public:
	void				draw();
	void				step(float dt=1.f/60);
	void				addSphere(Sphere& s){ m_sphere.push_back(s); }

};
#endif //PHYSICS_ENGINE
