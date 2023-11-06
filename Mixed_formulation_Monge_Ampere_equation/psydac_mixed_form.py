# -*- coding: UTF-8 -*-
#

from mpi4py import MPI
from sympy import pi, cos, sin, cosh, Tuple, sinh, log, exp, sqrt, atan2
from sympy.abc import x, y
import pytest
import os

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import inner, div, curl, cross
from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.topology import Square, NCube
from sympde.topology import Union
from sympde.expr     import linearize
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.api.essential_bc   import apply_essential_bc
from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.iterative_solvers import cg, pcg, bicg, lsmr

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
def scipy_solver(M, b):
    x  = spsolve(M.tosparse().tocsr(), b.toarray())
    x  = array_to_stencil(x, b.space)
    return x,0

#------------------------------------------------------------------------------
def psydac_solver(M, b):
    return lsmr(M, M.T, b, maxiter=10000, tol=1e-6)
    
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
    

#------------------------------------------------------------------------------
def run_non_linear_poisson(degree= [3, 3], ncells=[6, 6], comm=None, scipy=False):

    # Maximum number of Newton iterations and convergence tolerance
    N      = 1
    TOL    = 1e-8
    nbasis = [ncells[0]+degree[0], ncells[1]+degree[1]]   

    # Define topological domain
    Omega = Square() #Domain.from_file(filename)
    
    B_neumann = Omega.boundary
    nn = NormalVector('nn')

    # Method of manufactured solutions: define exact
    # solution phi_e, then compute right-hand side f
    x, y  = Omega.coordinates

    u_ex = 0.5*(x**2+y**2)

    # Define abstract model 
    V1 = VectorFunctionSpace('V1', Omega, kind='Hdiv')
    V2 = ScalarFunctionSpace('V2', Omega, kind='L2')
    # ...
    u0, u, v, du = elements_of(V1, names='u0, u, v, du')
    p0, p, q, dp = elements_of(V2, names='p0, p, q, dp')
    
    #.. Test 1
    #g = 1.6246394976381604/(1+5*exp(-50* abs((dx1(u)-0.5)**2+(dx2(u)-0.5)**2-0.09)))
    #rho = 1+5*exp(-50* abs((x-0.5)**2+(y-0.5)**2-0.09))

    #.. Test 2
    #g = 3.553841799466826/(1.+ 9./(1.+(10.*sqrt((dx1(u)-0.7-0.25*0.)**2+(dx2(u)-0.5)**2)*cos(atan2(dx2(u)-0.5,dx1(u)-0.7-0.25*0.) -20.*((dx1(u)-0.7-0.25*0.)**2+(dx2(u)-0.5)**2)))**2) )
    #rho = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(atan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2)

    #.. Test 3
    g   = 0.5698752034687177/( 1./(2.+cos(8.*pi*sqrt((u[0]-0.5)**2+(u[1]-0.5)**2))))
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

    f  = sqrt( dx1(u[0])**2+dx2(u[1])**2+2.*dx1(u[1])**2 + 2.*g )
    #.. Assembles nonlinear Poisson
    l  = LinearForm( (v, q), integral(Omega, f*q - div(u0)*q - dot(u0,v) ))

    #... Linear model (for Picard iteration)
    a = BilinearForm( ((du,dp),(v, q)), integral(Omega, dot(du, v) + div(du) * q + dp * div(v) ) )

    # Boundaries
    boundary_0h = Union(*[Omega.get_boundary(**kw) for kw in get_boundaries(1,2)])
    boundary_1h = Union(*[Omega.get_boundary(**kw) for kw in get_boundaries(3,4)])
    ue         = Tuple(0, 0)
    bc          = [EssentialBC(du, ue,B_neumann)] #, EssentialBC(du, 0., boundary_1h)]
    
    equation = find((du, dp), forall=(v, q), lhs=a((du, dp), (v, q)), rhs=l(v, q), bc = bc)

    #... Define (abstract) norms
    Quality_mesh = Norm( sqrt((u[0] - x)**2 +(u[1] - y)**2), Omega, kind='l2')
    l2norm_err   = Norm((dx1(u[0])*dx2(u[1])-dx1(u[1])**2) - g, Omega, kind='l2')
    l2norm_res   = Norm(p, Omega, kind='l2')

    #... Create computational domain from topological domain
    Omega_h = discretize(Omega, ncells=ncells, comm=comm)

    # ... discrete spaces
    V1h = discretize(V1, Omega_h, degree=degree)
    V2h = discretize(V2, Omega_h, degree=degree)
    Xh  = V1h*V2h

    #... Discretize equation (u is free parameter and must be provided later)
    equation_h = discretize(equation, Omega_h, [Xh, Xh])#, backend=PSYDAC_BACKEND_GPYCCEL)

    a_h        = equation_h.lhs
    l_h        = equation_h.rhs
    
    #... Discretize norms
    Quality_mesh_h = discretize(Quality_mesh, Omega_h, V1h)
    l2norm_err_h   = discretize(l2norm_err, Omega_h, V1h)
    l2norm_res_h   = discretize(l2norm_res, Omega_h, V2h)

    # First guess: zero solution
    u_h  = FemField(V1h)
    u0h  = FemField(V1h)
    p_h  = FemField(V2h)
    dp_h = FemField(V2h)
    
    import numpy as np

    u0h[0].coeffs[nbasis[0]-1,0:nbasis[1]-1] = 1.
    u0h[1].coeffs[0:nbasis[0]-1,nbasis[1]-1] = 1.

    print(u0h[0].coeffs[0,0], u0h[0].coeffs[:,:], u0h[1].coeffs[0,nbasis[1]-1])  
    
    solver = scipy_solver if scipy else psydac_solver
    #... Picard iteration
    for n in range(N):

        M = a_h.assemble()
        b = l_h.assemble(u=u_h, u0 = u0h)
        print(M.toarray())

        apply_essential_bc(M, *equation_h.bc, identity=True)
        apply_essential_bc(b, *equation_h.bc)
        print(M.toarray()[nbasis[0]:2*nbasis[0],0:nbasis[1]-1])
        print(b.toarray()[:])
        
        x,info = solver(M, b)

        u_h[0].coeffs[:]             = x[0][:]
        u_h[1].coeffs[:]             = x[1][:]
        p_h.coeffs[:]                = x[2][:]

        print(u_h[0].coeffs[:])
        # ...
        u_h[0].coeffs[0,0:nbasis[1]-1]           = 0. 
        u_h[0].coeffs[nbasis[0]-1,0:nbasis[1]-1] = 1. 

        u_h[1].coeffs[0:nbasis[0]-1,0]           = 0.
        u_h[1].coeffs[0:nbasis[0]-1,nbasis[1]-1] = 1.
        # ...
        '''
        x = equation_h.solve( u = u_h, u0 = u0h)
        u_h[0].coeffs[:] = x[0].coeffs[:]
        u_h[1].coeffs[:] = x[1].coeffs[:]
        # ... update the mapping's boudnary
        u_h[0].coeffs[nbasis[0]-1,:] = 1. 
        u_h[1].coeffs[:,nbasis[1]-1] = 1.

        p_h.coeffs[:]    = x[2].coeffs[:]
        '''
        # ...
        
        dp_h.coeffs[:]  -= p_h.coeffs[:]
        res_l2 = l2norm_res_h.assemble(p = dp_h)
        dp_h.coeffs[:]   = p_h.coeffs[:]

        print('==== iteration number = {} res_l2 = {} ===='.format(n+1, res_l2))

        if res_l2 <= TOL:
            print('CONVERGED')
            break

    print()
    print('==== iteration number = {} --> l2_res(u_k+1-u_k) = {}'.format(n+1, res_l2))

    # Quantity of displacement
    Qual_mesh = Quality_mesh_h.assemble(u = u_h)
    print('l1 norm of grad(u)-Id = {}'.format(Qual_mesh))

    # Computes l2 error norm in adpted map
    l2_error = l2norm_err_h.assemble(u=u_h)
    print('l2 norm of det(Hes(u_h)-f = {}'.format(l2_error))

    return res_l2, u_h

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_nonlinear_MFMAE_square():

    res_l2, u_h = run_non_linear_poisson()


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
             density[i,j]= 1./(2.+cos(8.*pi*sqrt((I[i]-0.5)**2+(J[j]-0.5)**2)))
             sX[i,j] = u_h[0](I[i],J[j]) 
             sY[i,j] = u_h[1](I[i],J[j])

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
    #axes[0].margins(0,0)
    #axes[0].axis('off')
        
    axes[1].set_title( 'Density function')
    im = axes[1].contourf(X, Y, density.T, cmap= 'jet')
    divider = make_axes_locatable(axes[1]) 
    cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('meshes_examples.png')
    plt.show()
    
    expected_l2_residual =  1e-6
    assert( abs(res_l2 - expected_l2_residual) < 1.e-6)

test_nonlinear_MFMAE_square()

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
