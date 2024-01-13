from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


from sympy import pi, cos, sin, exp, log, symbols, sqrt
from sympy.utilities.lambdify import implemented_function
import pytest

from sympde.calculus import grad, dot
from sympde.calculus import laplace, div
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Square
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.expr.expr import linearize
from sympde.core     import Constant

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

x,y,z = symbols('x1, x2, x3')

from sympy import diff
dx  = lambda e: diff(e,x)
dy  = lambda e: diff(e,y)

from numpy import random, max, absolute
import numpy as np

#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)

from matplotlib import pyplot as plt
#from simplines import plot_field_2d
from psydac.utilities.utils import refine_array_1d
import numpy as np
levels = np.linspace(-0.,1.,100)

def plot_field(field, N=40, i_sav = 0):
    Vh = field.space
    eta1 = refine_array_1d( Vh.spaces[0].breaks, N )
    eta2 = refine_array_1d( Vh.spaces[1].breaks, N )
    num = np.array( [[ field( e1,e2 ) for e2 in eta2] for e1 in eta1] )
    plt.contourf( eta1, eta2, num, levels, cmap='jet' )
    plt.colorbar()
    plt.savefig('figs/u_{}.png'.format(i_sav))
    #plt.show(block=False)
    plt.close()
def plot_res(n_itime, dt_evol, Sol_CH, GL_fe):
    # ...
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.subplot(121)
    plt.plot(n_itime, dt_evol, '*-b', linewidth = 2., label='$\mathbf{time-step}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('figs/dt.png')
    #plt.show(block=False)
    plt.close()
    # ---
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.subplot(121)
    plt.plot(n_itime, Sol_CH, 'o-k', linewidth = 2., label='$\mathbf{2th-Statistical-Moment}$')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('figs/M2.png')
    #plt.show(block=False)
    plt.close()
    # ---
    plt.figure()
    plt.axes().set_aspect('equal')
    #plt.subplot(122)
    plt.plot(n_itime, GL_fe,  '--or', label = '$\mathbf{GL-free-energy}$' )
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('figs/GlE.png')
    #plt.show(block=False)
    plt.close()


# ..Topological domain
domain         = Square()#bounds1=(0.,2.*pi**2), bounds2=(0.,2.*pi**2))
#B_dirichlet_0 = domain.boundary
# ..Function Space
V  = ScalarFunctionSpace('V', domain)


#  ... Parameters for generalized-alpha method
alpha_m = Constant(name='alpha_m') #0.5 * ((3. - rho_inf)/(1. + rho_inf))
alpha_f = Constant(name='alpha_f') #1/(1. + rho_inf)
gamma   = Constant(name='gamma') #0.5 + alpha_m - alpha_f
alpha   = 3000.
theta   = 3./2
# .. Defining the Linear form $G$
u   = element_of(V, name='u')
v   = element_of(V, name='v')
w   = element_of(V, name='w')

# time step
t   = Constant(name='t')
dt  = Constant(name='dt')
u0  = element_of(V, name='u0')
du0 = element_of(V, name='du0')



# Bilinear form a: V x V --> R
expr11 = ((3.*alpha/(2.*theta))*(1- 4.*theta*u0*(1.-u0)) + (1.-2.*u0)*laplace(u0) )*dot(grad(u), grad(v))
expr12 = ( (-6.*alpha)*(1.-2.*u0)*u - 2.*u*laplace(u0) + (1.-2.*u0)*laplace(u) ) * dot(grad(u0),grad(v))
#___
expr21 = u0 * (1.-u0) * laplace(u) * laplace(v)
expr22 = (1.-2.*u0)*u*laplace(u0)*laplace(v)
a = BilinearForm((u, v), integral(domain, alpha_m * u * v + (alpha_f * gamma * dt) * ( expr11  + expr21 + expr12 + expr22 ) ))
# Linear form l: V --> R
l = LinearForm(v, integral(domain,  du0 * v + ((3.*alpha/(2.*theta))*(1. - 4.*theta*u0*(1.-u0) ) + (1.-2.*u0) * laplace(u0) ) * dot(grad(u0),grad(v)) + u0 * (1. - u0) * laplace(u0) * laplace(v)  ))

# 2-order computes statistical moment
l_SM = LinearForm(v, integral(domain,  (u0-du0)**2*v ))

# ... Ginzburg–Landau free energy
l_FE = LinearForm(v, integral(domain,  ( u0*log(abs(u0)) +(1-u0)*log(abs(1-u0)) + 2*theta*u0*(1.-u0) + theta/(3.*alpha)*sqrt(dx(u0)**2+dy(u0)**2) ) * v ))


# Variational model
equation = find(u, forall=v, lhs=a(u, v), rhs=-l(v) )


# Create computational domain from topological domain
domain_h = discretize(domain, ncells=[16,16])

# Discrete spaces
Vh = discretize(V, domain_h, degree=[2,2], periodic = [True, True])

# Discretize equation using Dirichlet bc
equation_h = discretize(equation, domain_h, [Vh, Vh], periodic = [True, True])

# .. Computes Residual
lh     = discretize(l, domain_h, Vh)

# 2th-order computes statistical moment
lh_sm  = discretize(l_SM, domain_h, Vh)

# ... Ginzburg–Landau free energy
lh_fe  = discretize(l_FE, domain_h, Vh)

# ...
nbasis = [w.nbasis for w in Vh.spaces]
p1,p2  = Vh.degree


def G_alpha_solve(alpha_mh, alpha_fh, gammah, dt_h, u0_h, du0_h, niter=50):

    Un_f    = FemField( Vh, Vh.vector_space.zeros() )
    Un_m    = FemField( Vh, Vh.vector_space.zeros() )
    # ... tools for G-alpha method
    Un      = FemField( Vh, Vh.vector_space.zeros() )
    dUn     = FemField( Vh, Vh.vector_space.zeros() )
    # ...
    Un.coeffs._data[p1:-p1,p2:-p2]  = u0_h.coeffs._data[p1:-p1,p2:-p2]
    Un.coeffs.update_ghost_regions()
    dUn.coeffs._data[p1:-p1,p2:-p2] = (gammah-1.)/gammah * du0_h.coeffs._data[p1:-p1,p2:-p2]
    dUn.coeffs.update_ghost_regions()
    #...Newton iteration for non-linear system
    for i in range(niter):
        #... alpha level
        Un_m.coeffs._data[p1:-p1,p2:-p2] = du0_h.coeffs._data[p1:-p1,p2:-p2] + alpha_mh *(dUn.coeffs._data[p1:-p1,p2:-p2]- du0_h.coeffs._data[p1:-p1,p2:-p2])
        Un_m.coeffs.update_ghost_regions()
        Un_f.coeffs._data[p1:-p1,p2:-p2] = u0_h.coeffs._data[p1:-p1,p2:-p2] + alpha_fh *(Un.coeffs._data[p1:-p1,p2:-p2]- u0_h.coeffs._data[p1:-p1,p2:-p2])
        Un_f.coeffs.update_ghost_regions()

        delta_x  = equation_h.solve(u0 = Un_f, du0 = Un_m, dt = dt_h, alpha_m = alpha_mh, alpha_f = alpha_fh, gamma = gammah)
        # ...
        Un.coeffs._data[p1:-p1,p2:-p2]  = Un.coeffs._data[p1:-p1,p2:-p2] + gammah * dt_h * delta_x.coeffs._data[p1:-p1,p2:-p2]
        Un.coeffs.update_ghost_regions()
        dUn.coeffs._data[p1:-p1,p2:-p2] = dUn.coeffs._data[p1:-p1,p2:-p2] + delta_x.coeffs._data[p1:-p1,p2:-p2]
        dUn.coeffs.update_ghost_regions()

        # assemble the rhs and convert it to numpy array
        res = lh.assemble(u0=Un_f, du0=Un_m).toarray()
        Res        = max(absolute(res))

        if Res < 1e-4 :
            print(rank, 'perform the iteration number : = {} Residual  = {}'.format( i, Res))
            break
    return Un, dUn, Res

if rank == 0 :
	np.savetxt('x0.txt',  (1.-2.*random.rand(nbasis[0],nbasis[1]))*0.05 +0.63, fmt='%.10e')
	u0_h    = FemField( Vh, Vh.vector_space.zeros() )
	u0_h.coeffs._data[p1:-p1,p2:-p2] = np.loadtxt('x0.txt')
	u0_h.coeffs.update_ghost_regions()
else : 
	True

comm.Barrier()  # Synchronize both processors before starting the next iteration
if rank == 1:
	u0_h    = FemField( Vh, Vh.vector_space.zeros() )
	u0_h.coeffs._data[p1:-p1,p2:-p2] = np.loadtxt('x0.txt')
	u0_h.coeffs.update_ghost_regions()
	
#.. computes the projection of du_0 in the space
a_app = BilinearForm((u, v),integral(domain,u*v))
#..
l_app = LinearForm(v, integral(domain,  ( (3.*alpha/(2.*theta)) * (1.-4.*theta*u0*(1.-u0)) + (1.-2.*u0)*laplace(u0)) *dot(grad(u0), grad(v)) + u0*(1.-u0)*laplace(u0)*laplace(v)  ))

#..
equation_app = find(u, forall=v, lhs=a_app(u, v), rhs= -l_app(v))
#..
equation_app_h = discretize(equation_app, domain_h, [Vh, Vh], periodic = [True, True])
# Solve linear system
du0_h = equation_app_h.solve(u0 = u0_h)
du0_h.coeffs.update_ghost_regions()


Tf      = 1.
t_h     = 0.
Sol_CH  = []
GL_fe   = []
dt_evol = []
n_itime = []
dt_h    = 1e-8

#un, du, Sol_CH, GL_fe, dt_evol, n_itime, dt_h  = Time_dependent_Poisson(Tf, t_h, u0_h, du0_h, Sol_CH, GL_fe, dt_evol, n_itime, dt_h)
# ...
U0      = FemField( Vh, Vh.vector_space.zeros() )
U0.coeffs._data[p1:-p1,p2:-p2]  = u0_h.coeffs._data[p1:-p1,p2:-p2]

#  ... Parameters for generalized-alpha method
rho_infh = 0.5
alpha_mh = 0.5 * ((3. - rho_infh)/(1. + rho_infh))
alpha_fh = 1/(1. + rho_infh)
gammah   = 0.5 + alpha_mh - alpha_fh
# ...For t>t0
i_sav   = 0
i_refus = 0
plot_field(u0_h, N=2, i_sav = i_sav)

if size != 2:
    if rank == 0:
        print("This code requires exactly 2 processors.")
    MPI.Finalize()
    exit()
    
Un      = FemField( Vh, Vh.vector_space.zeros() )
dUn     = FemField( Vh, Vh.vector_space.zeros() )

while t_h < Tf :
	t_h += dt_h
	# ...
	if rank == 0:  # Processor 0 sends and receives data from Processor 1
		# ... G alpha method TODO RES ??
		Un, dUn, Res  = G_alpha_solve(alpha_mh, alpha_fh, gammah, dt_h, u0_h, du0_h)
		comm.send(Res, dest=1, tag = 3)
		comm.send(Un.coeffs._data[p1:-p1,p2:-p2], dest=1)
		comm.send(dUn.coeffs._data[p1:-p1,p2:-p2], dest=1, tag=2)
		dt_l          = dt_h
		dt_h          = comm.recv(source=1)
		ro_n          = comm.recv(source=1, tag=2)
		print(f"Processor 0 received data from Processor 1: {dt_h}")
	else:  # Processor 1 sends and receives data from Processor 0
		# ... Implicite,		
		BE_Un         = G_alpha_solve(1., 1., 1., dt_h, u0_h, du0_h)[0]
		# ... Algorithm for Time step
		Res           = comm.recv(source=0, tag = 3)
		Un.coeffs._data[p1:-p1,p2:-p2]  = comm.recv(source=0)
		Un.coeffs.update_ghost_regions()
		dUn.coeffs._data[p1:-p1,p2:-p2] = comm.recv(source=0, tag = 2)
		dUn.coeffs.update_ghost_regions()
		# ...
		if Res >1e-3:
			ro_n          = 1.
			dt_l          = dt_h
			dt_h          = 5e-8			
		else :
			norm_dif      = np.sqrt(np.sum(lh_sm.assemble(u0=Un, du0=BE_Un).toarray()))
			norm_sol      = np.sqrt(np.sum(lh_sm.assemble(u0=2.*Un, du0=Un).toarray()))
			ro_n          = norm_dif/norm_sol
			dt_l          = dt_h
			dt_h          = 0.9*(1e-3/ro_n)**(1/2) * dt_h
		comm.send(dt_h, dest=0)
		comm.send(ro_n, dest=0, tag=2)
		print(f"Processor 1 received data from Processor 0: {dt_h}")
	comm.Barrier()  # Synchronize both processors before starting the next iteration
	print('rank', rank, 'ro_n=', ro_n,'computed dt before refusion', dt_h,'\n\n\n')

	while(ro_n>1e-3):
		t_h -= dt_l
		t_h += dt_h
		if rank == 0:  # Processor 0 sends and receives data from Processor 1
			# ... G alpha method
			Un, dUn, Res  = G_alpha_solve(alpha_mh, alpha_fh, gammah, dt_h, u0_h, du0_h)
			# ...
			comm.send(Res, dest=1, tag = 3)
			comm.send(Un.coeffs._data[p1:-p1,p2:-p2], dest=1)
			comm.send(dUn.coeffs._data[p1:-p1,p2:-p2], dest=1, tag=2)
			dt_l          = dt_h
			dt_h          = comm.recv(source=1)
			ro_n          = comm.recv(source=1, tag=2)
		else:  # Processor 1 sends and receives data from Processor 0
			i_refus += 1
			# ... Implicite,		
			BE_Un         = G_alpha_solve(1., 1., 1., dt_h, u0_h, du0_h)[0]
			# ... Algorithm for Time step
			Res           = comm.recv(source=0, tag = 3)
			Un.coeffs._data[p1:-p1,p2:-p2] = comm.recv(source=0)
			Un.coeffs.update_ghost_regions()
			dUn.coeffs._data[p1:-p1,p2:-p2] = comm.recv(source=0, tag = 2)
			dUn.coeffs.update_ghost_regions()
			# ...
			if Res >1e-3:
				ro_n          = 1.
				dt_l          = dt_h
				dt_h          = 5e-8			
			else :
				norm_dif      = np.sqrt(np.sum(lh_sm.assemble(u0=Un, du0=BE_Un).toarray()))
				norm_sol      = np.sqrt(np.sum(lh_sm.assemble(u0=2.*Un, du0=Un).toarray()))
				ro_n          = norm_dif/norm_sol
				dt_l          = dt_h
				dt_h          = 0.9*(1e-3/ro_n)**(1/2) * dt_h
			comm.send(dt_h, dest=0)
			comm.send(ro_n, dest=0, tag=2)
		print('rank=', rank, 'ro_n=', ro_n, 'computed dt after refusion', dt_h,'\n\n')
		comm.Barrier()  # Synchronize both processors before starting the next iteration
	comm.Barrier()  # Synchronize both processors before starting the next iteration
	# .. update u0
	u0_h.coeffs._data[p1:-p1,p2:-p2]  = Un.coeffs._data[p1:-p1,p2:-p2]
	u0_h.coeffs.update_ghost_regions()
	du0_h.coeffs._data[p1:-p1,p2:-p2] = dUn.coeffs._data[p1:-p1,p2:-p2]
	du0_h.coeffs.update_ghost_regions()

	if rank == 1:
		i_sav +=1
		plot_field(u0_h, N=2, i_sav = i_sav)
		# ...
		dt_evol.append(np.exp(float(np.format_float_scientific( dt_l, unique=False, precision=2)))-1.)
		n_itime.append(np.exp(float(np.format_float_scientific( t_h, unique=False, precision=2)))-1.)
		Sol_CH.append(np.sum(lh_sm.assemble(u0=U0, du0=u0_h).toarray()))
		GL_fe.append(np.sum(lh_fe.assemble(u0=u0_h).toarray()))
		print('time step = ', t_h,'i_sav', i_sav, 'i_refus', i_refus)
		plot_res(n_itime, dt_evol, Sol_CH, GL_fe)
		np.savetxt('SM.txt', dt_evol, fmt='%.10e')
		np.savetxt('SM.txt', Sol_CH, fmt='%.10e')
		np.savetxt('GL_free_Adenergy.txt', GL_fe, fmt='%.10e')
		np.savetxt('x.txt', u0_h.coeffs._data[p1:-p1,p2:-p2], fmt='%.10e')
		np.savetxt('dx.txt', du0_h.coeffs._data[p1:-p1,p2:-p2], fmt='%.10e')
		
