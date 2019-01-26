def _mix_all_replicas_weave(nstates, replica_states, u_kl, Nij_proposed, Nij_accepted, verbose=True):
    """

    CODE AND FILE TO BE REMOVED IN UPCOMING VERSION

    Attempt exchanges between all replicas to enhance mixing.
    Acceleration by 'weave' from scipy is used to speed up mixing by ~ 400x.

    """

    # TODO: Replace this with a different acceleration scheme to achieve better performance?

    # Determine number of swaps to attempt to ensure thorough mixing.
    # TODO: Replace this with analytical result computed to guarantee sufficient mixing.
    # TODO: Alternatively, use timing to figure out how many swaps we can do and still keep overhead to ~ 1% of iteration time?
    # nswap_attempts = nstates**5 # number of swaps to attempt (ideal, but too slow!)
    nswap_attempts = nstates**4 # number of swaps to attempt
    # Handled in C code below.

    if verbose: print("Will attempt to swap all pairs of replicas using weave-accelerated code, using a total of %d attempts." % nswap_attempts)

    from scipy import weave

    # TODO: Replace drand48 with numpy random generator.
    code = """
    // Determine number of swap attempts.
    // TODO: Replace this with analytical result computed to guarantee sufficient mixing.        
    //long nswap_attempts = nstates*nstates*nstates*nstates*nstates; // K**5
    //long nswap_attempts = nstates*nstates*nstates; // K**3
    long nswap_attempts = nstates*nstates*nstates*nstates; // K**4

    // Attempt swaps.
    for(long swap_attempt = 0; swap_attempt < nswap_attempts; swap_attempt++) {
        // Choose replicas to attempt to swap.
        int i = (long)(drand48() * nstates);
        int j = (long)(drand48() * nstates);

        // Determine which states these resplicas correspond to.
        int istate = REPLICA_STATES1(i); // state in replica slot i
        int jstate = REPLICA_STATES1(j); // state in replica slot j

        // Reject swap attempt if any energies are nan.
        if ((std::isnan(U_KL2(i,jstate)) || std::isnan(U_KL2(j,istate)) || std::isnan(U_KL2(i,istate)) || std::isnan(U_KL2(j,jstate))))
           continue;

        // Compute log probability of swap.
        double log_P_accept = - (U_KL2(i,jstate) + U_KL2(j,istate)) + (U_KL2(i,istate) + U_KL2(j,jstate));

        // Record that this move has been proposed.
        NIJ_PROPOSED2(istate,jstate) += 1;
        NIJ_PROPOSED2(jstate,istate) += 1;

        // Accept or reject.
        if (log_P_accept >= 0.0 || (drand48() < exp(log_P_accept))) {
            // Swap states in replica slots i and j.
            int tmp = REPLICA_STATES1(i);
            REPLICA_STATES1(i) = REPLICA_STATES1(j);
            REPLICA_STATES1(j) = tmp;
            // Accumulate statistics
            NIJ_ACCEPTED2(istate,jstate) += 1;
            NIJ_ACCEPTED2(jstate,istate) += 1;
        }

    }
    """

    # Stage input temporarily.
    nstates = nstates
    replica_states = replica_states
    u_kl = u_kl
    Nij_proposed = Nij_proposed
    Nij_accepted = Nij_accepted

    # Execute inline C code with weave.
    info = weave.inline(code, ['nstates', 'replica_states', 'u_kl', 'Nij_proposed', 'Nij_accepted'], headers=['<math.h>', '<stdlib.h>'], verbose=0,
                        extra_compile_args=['-w'] # inhibit compiler warnings
                        )

    # Store results.
    replica_states = replica_states
    Nij_proposed = Nij_proposed
    Nij_accepted = Nij_accepted

    return

