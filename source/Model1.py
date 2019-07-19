from pulp import *
import glob
import sys

def SolveModel1(fileName) :
    input = open(fileName, "r")

    # Parameter
    input.readline(); input.readline()

    tmpStr = input.readline()          # Get data line
    tmpStr = tmpStr[0:len(tmpStr) - 1] # Remove eol '\n'
    tmpLst = tmpStr.split("\t")        # Convert string to para list

    n       = int(tmpLst[0])   # Number of UTXO in UTXOs pool
    m       = int(tmpLst[1])   # Number of transaction output
    M       = int(tmpLst[3])   # Maximum size of transaction
    T       = int(tmpLst[5])   # Dust threshold
    alpha   = float(tmpLst[4]) # Fee rate = 1 satoshi = 10**(-8) BTC
    beta    = int(tmpLst[7])   # Size of new UTXO
    epsilon = int(tmpLst[6])   # Minimum of change output

    # Vin
    input.readline(); input.readline(); input.readline()

    V_u = list() # Set of UTXO’s values
    S_u = list() # set of transaction input size
    for _ in range(n) :
        tmpStr = input.readline()         # Get data line
        tmpStr = tmpStr[:len(tmpStr) - 1] # Remove eol '\n'
        tmpLst = tmpStr.split("\t")       # Convert string to para list
        
        V_u.append(int(tmpLst[2]))
        S_u.append(int(tmpLst[1]))

    # Vout
    input.readline(); input.readline(); input.readline()

    V_o = list() # Set of transaction output’s values
    S_o = list() # Set of transaction output’s size
    for _ in range(m) :
        tmpStr = input.readline()           # Get data line
        tmpStr = tmpStr[:len(tmpStr) - 1]   # Remove eol '\n'
        tmpLst = tmpStr.split("\t")         # Convert string to para list
        
        V_o.append(int(tmpLst[2]))
        S_o.append(int(tmpLst[1]))

    input.close()

    opt_model = LpProblem(name = "Model_1", sense = LpMinimize)

    # # * Declare variables * # #
    # Decision variables
    X = [LpVariable(name="x_{0}".format(i), cat = LpBinary) for i in range(n)]

    # Intermediate variables
    sigma = LpVariable(name = "sig", cat = LpBinary)
    # Size of change output
    z_s   = LpVariable(name = "z_s", lowBound = 0, cat = LpInteger)
    # A value of change output
    z_v   = LpVariable(name = "z_v", lowBound = 0, cat = LpContinuous)

    # # * Objective Function * # #
    y = lpDot(S_u, X) + lpSum(S_o) + z_s

    opt_model += y

    # # * Constraint * # #
    # A transaction size may not exceed maximum block data size
    opt_model += y <= M

    # A transaction must have sufficient value for consuming
    opt_model += lpDot(V_u, X) == lpSum(V_o) + alpha*y + z_v

    # All the transaction outputs must be higher than the dust threshold
    opt_model += lpSum(v) >= T

    # z_s = (z_v > epsilon)? beta : 0
    large = sys.maxsize
    opt_model += z_v + large*(1 - sigma) >= epsilon + 0.001
    opt_model += z_v - large*(sigma)     <= epsilon
    opt_model += z_s >= sigma*beta

    # opt_model.solve(solver = GLPK_CMD(msg=0))
    opt_model.solve()

    return value(opt_model.objective)
