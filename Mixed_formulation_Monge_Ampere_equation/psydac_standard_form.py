# -*- coding: UTF-8 -*-
#

from mpi4py import MPI
from sympy import pi, cos, sin, cosh, sinh, log, exp, sqrt, atan2
from sympy.abc import x, y
import pytest
import os

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.topology import Square, NCube
from sympde.topology import Union
from sympde.expr     import linearize
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

from sympde.topology import dx, dy, dz
from sympde.topology import dx1, dx2, dx3

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

#------------------------------------------------------------------------------
def run_non_linear_poisson(Omega, degree= [3, 3], ncells=[32, 32], comm=None):

    # Maximum number of Newton iterations and convergence tolerance
    N   = 30
    TOL = 1e-9

    B_neumann = Omega.boundary
    nn  = NormalVector('nn')

    # Method of manufactured solutions: define exact
    # solution phi_e, then compute right-hand side f
    x, y  = Omega.coordinates

    u_ex = 0.5*(x**2+y**2)

    # Define abstract model
    V = ScalarFunctionSpace('V', Omega)
    v = element_of(V, name='v')
    u = element_of(V, name='u')

    #.. Test 1
    #g = 1.6246394976381604/(1+5*exp(-50* abs((dx1(u)-0.5)**2+(dx2(u)-0.5)**2-0.09)))
    #rho = 1+5*exp(-50* abs((x-0.5)**2+(y-0.5)**2-0.09))

    #.. Test 2
    #g = 3.553841799466826/(1.+ 9./(1.+(10.*sqrt((dx1(u)-0.7-0.25*0.)**2+(dx2(u)-0.5)**2)*cos(atan2(dx2(u)-0.5,dx1(u)-0.7-0.25*0.) -20.*((dx1(u)-0.7-0.25*0.)**2+(dx2(u)-0.5)**2)))**2) )
    #rho = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(atan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2)

    #.. Test 3
    g   = 0.5698752034687177/( 1./(2.+cos(8.*pi*sqrt((dx1(u)-0.5)**2+(dx2(u)-0.5)**2))))
    rho = 1./(2.+cos(8.*pi*sqrt((x-0.5)**2+(y-0.5)**2)))

    #... Test 4
    #g = 1.4259290652600725/( 1.+ 5.*exp(-50.*abs((dx1(u)-0.5-0.25*cos(2.*pi*0.25))**2-(dx2(u)-0.5-0.5 *sin(2.*pi*0.25))**2- 0.01)))
    #rho =  1.+ 5.*exp(-50.*abs((x-0.5-0.25*cos(2.*pi*0.25))**2-(y-0.5-0.5 *sin(2.*pi*0.25))**2- 0.01))

    #... Test 5
    #gamma = 5.
    #lamb = 100.
    #t = 0.75
    #x0 = t ; y0 = 0.2 + 0.5 * t ; x1 = 1. - t ; y1 = 0.8 - 0.5 * t
    #g = 1.2120563972692635/( 1. + gamma/cosh(lamb * ( dx1(u) - x0 + dx2(u) - y0 )) + gamma/cosh(lamb * ( dx1(u) - x1 + dx2(u) - y1 )))
    #rho = 1. + gamma/cosh(lamb * ( x - x0 + y - y0 )) + gamma/cosh(lamb * ( x - x1 + y - y1 ))

    #.. rhs of MAE
    #f = -sqrt( dx(dx(u))**2+dy(dy(u))**2+2.*dx(dy(u))**2 + 2.*g )

    f = -sqrt( dx1(dx1(u))**2+dx2(dx2(u))**2+2.*dx1(dx2(u))**2 + 2.*g )
    l0 = LinearForm( v, integral(Omega, f*v ))
    
    #... Neumann boundary conditions
    l1 = LinearForm(v, integral(B_neumann, v * dot(grad(u_ex), nn)))

    #.. Assembles nonlinear Poisson
    l  = LinearForm(v, l0(v) + l1(v))

    du = element_of(V, name='du')
    #... Linear model (for Picard iteration)
    a = BilinearForm((du, v), integral(Omega, dot(grad(du), grad(v) ) + 1e-8 * du * v) )

    equation = find(du, forall=v, lhs=a(du, v), rhs= l(v) )

    #... Define (abstract) norms
    Quality_mesh = Norm(u - u_ex, Omega, kind='h1')
    l2norm_err = Norm((dx1(dx1(u))*dx2(dx2(u))-dx1(dx2(u))**2) - g, Omega, kind='l2')

    l2norm_res = Norm(u, Omega, kind='l2')
    H1norm_res = Norm(u, Omega, kind='h1')

    #... Create computational domain from topological domain
    Omega_h = discretize(Omega, ncells=ncells, comm=comm)

    #... Create discrete spline space
    Vh = discretize(V, Omega_h, degree=degree)

    #... Discretize equation (u is free parameter and must be provided later)
    equation_h = discretize(equation, Omega_h, [Vh, Vh]) #, backend=PSYDAC_BACKEND_GPYCCEL)

    #... Discretize norms
    Quality_mesh_h = discretize(Quality_mesh, Omega_h, Vh)
    l2norm_err_h = discretize(l2norm_err, Omega_h, Vh)

    l2norm_res_h = discretize(l2norm_res, Omega_h, Vh)
    H1norm_res_h = discretize(H1norm_res, Omega_h, Vh)

    # First guess: zero solution
    u_h = FemField(Vh)
    du_h = FemField(Vh)
 
    #... Picard iteration
    for n in range(N):
        u_h = equation_h.solve( u = u_h)
 
        alpha = u_h.coeffs.toarray().sum()/len(u_h.coeffs.toarray())
        #u_h -= alpha
        u_h.coeffs[:,:] -= alpha
        du_h -= u_h
        res_H1 = l2norm_res_h.assemble(u = du_h)
        res_l2 = l2norm_res_h.assemble(u = du_h)
        du_h = u_h

        if res_H1 <= TOL:
            print('CONVERGED')
            break

    print()
    print('==== iteration number = {} ===='.format(n+1))

    # Quantity of displacement
    Qual_mesh = Quality_mesh_h.assemble(u = u_h)
    print('l1 norm of grad(u)-Id = {}'.format(Qual_mesh))

    # Computes l2 error norm in adpted map
    l2_error = l2norm_err_h.assemble(u=u_h)
    print('l2 norm of det(Hes(u_h)-f = {}'.format(l2_error))
 
    #... Comptes residual
    print('--> l2_res(u_k+1-u_k) = {} \n--> H1_res(u_k+1-u_k) = {}'.format(res_l2,res_H1))

    return res_H1, u_h

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_nonlinear_poisson_square():
    # Define topological domain
    Omega = Square() #Domain.from_file(filename)
    x, y  = Omega.coordinates
    
    res_H1, u_h = run_non_linear_poisson(Omega)

    #.. Test 1
    #rho = 1+5*exp(-50* abs((x-0.5)**2+(y-0.5)**2-0.09))

    #.. Test 2
    #rho = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(atan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2)

    #.. Test 3
    rho = 1./(2.+cos(8.*pi*sqrt((x-0.5)**2+(y-0.5)**2)))

    #... Test 4
    #rho =  1.+ 5.*exp(-50.*abs((x-0.5-0.25*cos(2.*pi*0.25))**2-(y-0.5-0.5 *sin(2.*pi*0.25))**2- 0.01))

    #... Test 5
    #gamma = 5.
    #lamb = 100.
    #t = 0.75
    #x0 = t ; y0 = 0.2 + 0.5 * t ; x1 = 1. - t ; y1 = 0.8 - 0.5 * t
    #rho = 1. + gamma/cosh(lamb * ( x - x0 + y - y0 )) + gamma/cosh(lamb * ( x - x1 + y - y1 ))
    
    #.... Ploting
    import matplotlib.pyplot            as     plt
    from   matplotlib                   import ticker,cm
    from   mpl_toolkits.mplot3d         import axes3d
    from   mpl_toolkits.axes_grid1      import make_axes_locatable
    import numpy                        as np
    
    # ...
    nbpts = 80
    I=J=np.linspace(0,1,nbpts)
    X,Y=np.meshgrid(I,J)
    sX =np.zeros((len(I),len(J)),float)
    sY =np.zeros((len(I),len(J)),float)
    density=np.zeros((len(I),len(J)),float)
    for i in range(nbpts):
        for j in range(nbpts):
             density[i,j]= rho.subs({x:I[i],y:J[j]})
             sX[i,j], sY[i,j] = u_h.gradient(I[i],J[j])

    #+++++++++++++++++++++++++++
    ## Affichage des solutions
    #+++++++++++++++++++++++++++
    figtitle  = 'MAE_equation'
    fig, axes = plt.subplots( 1, 2, figsize=[12,5], num=figtitle )
    for ax in axes:
        ax.set_aspect('equal')
    
    axes[0].set_title( 'Adapted mesh ' )
    for i in range(nbpts):
        phidx = sX[:,i]
        phidy = sY[:,i]

        axes[0].plot(phidx, phidy, '-b', linewidth = 0.54)
    for i in range(nbpts):
        phidx = sX[i,:]
        phidy = sY[i,:]

        axes[0].plot(phidx, phidy, '-b', linewidth = 0.54)
    i     = 0
    phidx = sX[:,i]
    phidy = sY[:,i]
    axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
    i     = nbpts-1
    phidx = sX[:,i]
    phidy = sY[:,i]
    axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
    #''    
    i     = 0
    phidx = sX[i,:]
    phidy = sY[i,:]
    axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
    i     = nbpts-1
    phidx = sX[i,:]
    phidy = sY[i,:]
    axes[0].plot(phidx, phidy, '-r', linewidth = 2.)
    axes[0].margins(0,0)
    axes[0].axis('off')
        
    axes[1].set_title( 'Density function')
    im = axes[1].contourf(X, Y, density.T, cmap= 'jet')
    divider = make_axes_locatable(axes[1]) 
    cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('meshes_examples.png')
    plt.show()
    
    expected_H1_residual =  1e-7

    assert( abs(res_H1 - expected_H1_residual) < 1.e-7)
test_nonlinear_poisson_square()

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()
    
teardown_module()
