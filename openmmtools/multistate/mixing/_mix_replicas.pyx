cimport cython
from libc.math cimport exp, isnan
from libc.stdio cimport printf
cdef extern from 'stdlib.h' nogil:
    double drand48()

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef long _mix_replicas_cython(long nswap_attempts, long nstates, long[:] replica_states, double[:,:] u_kl, long[:,:] Nij_proposed, long[:,:] Nij_accepted) nogil:
    cdef long swap_attempt
    cdef long i, j, istate, jstate, tmp_state
    cdef double log_P_accept
    for swap_attempt in range(nswap_attempts):
        i = <long>(drand48()*nstates)
        j = <long>(drand48()*nstates)
        istate = replica_states[i]
        jstate = replica_states[j]
        if (isnan(u_kl[i, istate]) or isnan(u_kl[i, jstate]) or isnan(u_kl[j, istate]) or isnan(u_kl[j, jstate])):
            continue
        log_P_accept = - (u_kl[i, jstate] + u_kl[j, istate]) + (u_kl[j, jstate] + u_kl[i, istate])
        Nij_proposed[istate, jstate] +=1
        Nij_proposed[jstate, istate] +=1
        if(log_P_accept>=0 or drand48()<exp(log_P_accept)):
            tmp_state = replica_states[i]
            replica_states[i] = replica_states[j]
            replica_states[j] = tmp_state
            Nij_accepted[istate,jstate] += 1
            Nij_accepted[jstate,istate] += 1
    return 0
