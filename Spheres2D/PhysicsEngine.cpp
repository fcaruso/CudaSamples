#include "PhysicsEngine.h"
#define _USE_MATH_DEFINES 
#include <math.h>


//////////////
///  apply gravity
void PhysicsEngine::applyGravity(float g)
{
	for (int i=0; i<m_sphere.size(); i++ )
	{
		Sphere& s = m_sphere[i];
		s.m_a = vec2(0.f, g);
	}
}

//////////////////////////////////
/// simple verlet integration scheme
void PhysicsEngine::integrate(float dt)
{
	for (int i=0; i<m_sphere.size(); i++)
	{
		Sphere& s = m_sphere[i];
		vec2 new_x = 2.f*s.m_x - s.m_xprev + s.m_a*dt*dt;
		s.m_xprev = s.m_x;
		s.m_x = new_x;
		s.m_v = (s.m_x - s.m_xprev)/dt;
	}
}

void PhysicsEngine::manageCollisions(float dt)
{
	for (int i=0 ; i < m_sphere.size() -1 ; i++ )
	{
		Sphere& s = m_sphere[i];

		// check collision with left
		float compenetration = (s.m_x.x - s.m_radius ) - m_box.m_min.x;
		
		if ( compenetration < 0.f )
		{
			s.m_xprev = s.m_x;
			s.m_x += vec2( compenetration, 0);
			s.m_v = glm::normalize( s.m_x - s.m_xprev ) * sqrtf ( glm::dot ( s.m_v, s.m_v ) );

		}

		// check collision with bottom
		compenetration = ( s.m_x.y - s.m_radius );

		if ( compenetration < 0.f )
		{
			s.m_xprev = s.m_x;
			s.m_x += vec2( 0, compenetration);
			s.m_v = glm::normalize( s.m_x - s.m_xprev ) * sqrtf ( glm::dot ( s.m_v, s.m_v ) );

		}

		// check collision with right
		compenetration = m_box.m_max.x - ( s.m_x.x + s.m_radius );

		if ( compenetration < 0.f )
		{
			s.m_xprev = s.m_x;
			s.m_x -= vec2( compenetration, 0);
			s.m_v = glm::normalize( s.m_x - s.m_xprev ) * sqrtf ( glm::dot ( s.m_v, s.m_v ) );

		}

		//for (int j=i ; j < m_sphere.size() ; j++ )
		//{
		//	Sphere& s1 = m_sphere[i];
		//	Sphere& s2 = m_sphere[j];
		//	vec2	d = s2.m_x - s1.m_x;
		//	float distance = sqrtf(glm::dot(d,d));

		//	float compenetration = distance - ( s1.m_radius + s2.m_radius );

		//	if ( compenetration < 0.f )
		//	{
		//		float m1_plus_m2 = 1./s1.m_im + 1./s2.m_im;
		//		float delta_x1 = compenetration * (1.f / s2.m_im ) / m1_plus_m2;
		//		float delta_x2 = compenetration - delta_x1;

		//		vec2 u = glm::normalize(d);

		//		s1.m_xprev = s1.m_x;
		//		s2.m_xprev = s2.m_x;

		//		s1.m_x += delta_x1 * u;
		//		s2.m_x += delta_x2 * u;

		//		s1.m_v = sqrtf(glm::dot(s1.m_v , s1.m_v)) * u;
		//		s2.m_v = sqrtf(glm::dot(s2.m_v , s2.m_v)) * u;
		//	}
		//}
	}
}