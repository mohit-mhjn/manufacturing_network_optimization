def solve_model(model):       # Custom Solve Method
    import datetime
    trial_pool = {
                    'sample1':{'solution_file':None,'aspiration':None},
                    'sample2':{'solution_file':None,'aspiration':None},
                    'sample3':{'solution_file':None,'aspiration':None}
                 }

    #model1 = copy.deepcopy(model)
    #model2 = copy.deepcopy(model)
    #model3 = copy.deepcopy(model)

    print ("job initiated! \n loading solver......")

    mip_gap = 0.05
    solver_timeout = 300
    number_of_trials = 1
    solver_sh = 'cplex'  #initialization Setting
    engage_neos = True  #initialization Setting
    j = 1
    timeout_arguments = {'cplex':'timelimit','cbc':'sec'}
    gap_arguments = {'cplex':'mipgap','cbc':'ratio'}   # Cplex Local Executable Take : "mip_tolerance_mipgap", mipgap is for neos version

    # NEOS Server Library Dependency
    # pyro4
    # suds
    # openopt

    while j < number_of_trials + 1:
        from pyomo.opt import SolverFactory, SolverManagerFactory
        #model.pprint()
        opt = SolverFactory(solver_sh, solver_io = 'lp')
        print ('\ninterfacing solver shell :',solver_sh)
        if engage_neos:
            solver_manager = SolverManagerFactory('neos')
        opt.options[timeout_arguments[solver_sh]]= solver_timeout
        opt.options[gap_arguments[solver_sh]] = mip_gap
        #opt.symbolic_solver_labels=True
        #opt.options['slog'] = 1
        #opt.enable = 'parallel'
        print ('\tsolver options >> \n\n\tTolerance Limits:\n\tmip_gap = %s \n\ttimeout = %s'%(str(mip_gap),str(solver_timeout)))
        print ("\nProcessing Trial Number :",j)
        print ("\nJob Triggered at :",str(datetime.datetime.now()))
        print ('\ncalculating solution...... !! please wait !!    \n to interrupt press ctrl+C\n')
        try:
            if engage_neos:
                results = solver_manager.solve(model,opt = opt, tee= True)
            else:
                opt.options['threads'] = 3
                results = opt.solve(model) #,tee=True) # Method Load Solutions is not available in pyomo versions less than 4.x
        except:
            j = j+1
            mip_gap = (j-1)*mip_gap
            solver_sh = 'cplex'
            engage_neos = True
            continue
        #results.write(filename='results'+str(datetime.date.today())+'.json',format = 'json')
        #print (results)
        if str(results['Solver'][0]['Termination condition']) in ['infeasible','maxTimeLimit','maxIterations','intermediateNonInteger']:
            j = j+1
            mip_gap = (j-1)*mip_gap
            solver_sh = 'cplex'
            engage_neos = True
            if j == number_of_trials + 1:
                #print (results['Problem'])
                #print (results['Solver'])
                raise AssertionError("Solver Failed with Termination Status : %s"%(str(results['Solver'][0]['Termination condition'])))
                exit(0)
            print ('Terminated by:',str(results['Solver'][0]['Termination condition']))
            print ("\n\nRetrying...\n\n")
        else:
            print ("solution captured")
            model.solutions.store_to(results)
            #post_process_results()
            break

    print (results['Problem'])
    print (results['Solver'])
    print ("\nSolution Retrived at:",str(datetime.datetime.now()))
    return [model,results]
