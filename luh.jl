"""
    Light up puzzle

    The table is such that
     blank cells are coded with 6
     unrestricted black cells with 5
     black cells restricted from 0 to 4 with the corresponding number.

     Then the light up table in Emílio’s paper
     2007 IEEE Congress on Evolutionary Computation (CEC 2007) page 1404 is

    6 6 6 6
    6 3 6 2
    6 6 6 6
    6 1 6 5
"""

# The problems are from
# https://www.minijuegos.com/juegos/jugar.php?id=3502
# https://www.puzzle-light-up.com/
# http://www.nikoli.co.jp/en/puzzles/light_up/


using DelimitedFiles, OffsetArrays, Crayons, LinearAlgebra, Statistics, PyPlot, Printf
pygui(true)

# Constants definition

const BLACK0 = 0    # Black cell with restriction 0
const BLACK1 = 1    # Black cell with restriction 1
const BLACK2 = 2    # Black cell with restriction 2
const BLACK3 = 3    # Black cell with restriction 3
const BLACK4 = 4    # Black cell with restriction 4
const BLACK5 = 5    # Black whithout restriction
const WHITE  = 6    # white cell
const BULB   = 7    # cell with bulb

# More constants

NONE 		= 0	# If INITBULBS = NONE there are no bulbs at initialization
FULL 		= 1	# if INITBULBS = FULL all cell are initiated with bulbs
RANDOM		= 2	# Random disposition of bulbs or random update of neurons

INITBULBS 	= RANDOM	# Initialization of bulbs. May be NONE, FULL or RANDOM
UPDATE 		= RANDOM 	# Update of neurons

PRINTLU 	= true
STEPSTOPRINT = 1000
DOMEANS = false

# Auxiliary functions
len(x) = length(x)

# Create an array of type T and initialized with x
function CreateArray(x::T,n) where T
    A = Array{T}(undef,n)
    for k in 1:n
        A[k] = x
    end
    A
end

function CreateArray(x::T,m,n) where T
    A = Array{T}(undef,m,n)
    for k in 1:m, l in 1:n
        A[k,l] = x
    end
    A
end

# Return true if cell (i,j) is inside the table
# and false otherwise
function valid(i,j,M,N)
    1 <= i <= M && 1 <= j <= N ? true : false
end

# Sign function
function sgn(x)
    if x ≥ 0 return 1 end
    0
end

# Other type of sgn function
function sgn(x,s)
   if x > 1 return 1 end
   if x < 0 return 0 end
   s
end

# return a randon number i in an array r
# and the new array r without i
function randx(r::Array{Int})
    i = rand(r)
    if length(r) > 1
        ir = findall(x->x!=i,r)
        r = r[ir]
    else
        r = []
    end
    i, r
end

# Matlab's tic and toc 
function tic()
   global _Time0 = time_ns()/1e9
   0
end

function toc()
   time_ns()/1e9 - _Time0
end


# Experimental way to calculate the network convergence rate
function expnetρ(W,J)
	N = len(J)
	nΔE = zeros(N)
	aΔE = zeros(N)
   aρ = zeros(N)
	for i in 1:N
      for n in 1:1000
         X = [rand([0,1]) for k in 1:N]
			ΔE = -W[i,i]/2 - abs(H(X,W,J)[i])
			if ΔE < 0 nΔE[i] = nΔE[i] - ΔE end
			aΔE[i] = aΔE[i] + abs(ΔE)
		end
		aρ[i] = nΔE[i]/aΔE[i]
	end
	mean(aρ)
end


# Set some constants.
# Number of lines, number of columns
# number of neurons and number of black cells
function LUConstants(LightUpTable)
    M,N = size(LightUpTable)
    nB = 0
    for lin in 1:M, col in 1:N
        if LightUpTable[lin,col] < 6
            nB = nB + 1
        end
    end
    nW = M*N-nB
    M, N, nW, nB
end


# Auxiliary functions for printing
function FillBorder(A,P)
	M, N = size(A)
	M = M - 2
	N = N - 2
   BorderAtribute = Crayon(foreground=:blue,background=:black)

    A[0,0] = BorderAtribute; P[0,0] = "\u2588"
    A[0,N+1] = BorderAtribute; P[0,N+1] = "\u2588"
    A[M+1,0] = BorderAtribute; P[M+1,0] = "\u2588"
    A[M+1,N+1] = BorderAtribute; P[M+1,N+1] = "\u2588"
    for n in 1:N
        A[0,n] = BorderAtribute
        P[0,n] = "\u2588"       # upper border
    end
    for n in 1:N
        A[M+1,n] = BorderAtribute
        P[M+1,n] = "\u2588"     # lower border
    end
    for m in 1:M
        A[m,0] = BorderAtribute
        P[m,0] = "\u2588"            # left border
    end
    for m in 1:M
        A[m,N+1] = BorderAtribute
        P[m,N+1] = "\u2588"          # right border
    end
end

function FillBlacks(A,P,LightUpTable)
    M,N=size(LightUpTable)
    ablacks = ["0","1","2","3","4"," "]
    BlacksAtribute = Crayon(foreground=:white, background=:red)
    for m in 1:M, n in 1:N
        if LightUpTable[m,n] < WHITE
            A[m,n] = BlacksAtribute
            P[m,n] = ablacks[LightUpTable[m,n]+1]
        end
    end
end

function FillBulbs(X,A,P,NeuronToCell)
    BulbAtribute = Crayon(foreground=:black,background=:yellow)
    nW = len(NeuronToCell)
    for n in 1:nW
        i,j = NeuronToCell[n][2]
        if X[n] == 1
            A[i,j] = BulbAtribute
            P[i,j] = "*"      # \U1f4a1   bulb
        end
    end
end


function PrintTable(A,P)
	M,N = size(A)
	M = M - 2
	N = N - 2
    Crayons.force_color(true)
    for m in 0:M+1
        println(Crayon(reset=true))
        for n in 0:N+1 print(A[m,n],P[m,n]) end
    end
    A, P
end


function PrintLightUp(X,LightUpTable=LightUpTable,NeuronToCell=NeuronToCell)
    nW = len(X)
    M,N=size(LightUpTable)
    WhitesAtribute = Crayon(foreground=:white,background=:black)
    A = CreateArray(WhitesAtribute,M+2,N+2)
    A = OffsetArray(A,0:M+1,0:N+1)
    P = CreateArray("\u2588",M+2,N+2)
    P = OffsetArray(P,0:M+1,0:N+1)
    FillBorder(A,P)
    FillBlacks(A,P,LightUpTable)
    FillBulbs(X,A,P,NeuronToCell)
    PrintTable(A,P)
    nothing
end


# Returns the vector with the initial values of neurons X, the list
# of correspondence between the numbering of neurons and cells, NeuronToCell
# and the correspondence matrix between cells and neurons, CellToNeuron
function Neuron(LightUpTable)
   M,N,nW,nB = LUConstants(LightUpTable)
   X = CreateArray(0,M*N-nB)
   CellToNeuron = CreateArray(0,M,N)
   NeuronToCell = []
   n = 1
   for l in 1:M, c in 1:N
		if LightUpTable[l,c] == WHITE
			if INITBULBS == FULL
				X[n] = 1
            elseif INITBULBS == NONE
				X[n] = 0
			else
				X[n] = rand([0,1])     # put bulbs at random
			end
         push!(NeuronToCell,(n,(l,c)))
         CellToNeuron[l,c] = n
         n = n + 1
      else
			CellToNeuron[l,c] = LightUpTable[l,c]
      end
   end
   X, NeuronToCell, CellToNeuron
end


# Return 1 if a bulb in neuron i shines neuron j
# otherwise return 0. If i==j return 0
function Light(i,j,LightUpTable=LightUpTable,NeuronToCell=NeuronToCell)
    if i == j return 0 end          # same neuron
    l1,c1 = NeuronToCell[i][2]      # take the cartesian coordinates of neuron i
    l2,c2 = NeuronToCell[j][2]      # take the cartesian coordinates of neuron j
    if l1 != l2 && c1 != c2 return 0 end
    if l1 == l2                     # they are in the same line
        c1,c2 = c1<c2 ? (c1,c2) : (c2,c1)
        for k in c1+1:c2
            if LightUpTable[l1,c1] != LightUpTable[l1,k] return 0 end
        end
    end
    if c1 == c2                     # they are in the same column
        l1,l2 = l1<l2 ? (l1,l2) : (l2,l1)
        for k in l1+1:l2
            if LightUpTable[l1,c1] != LightUpTable[k,c1] return 0 end
        end
    end
    1
end

# Return the array of black cells values
# near the neuron k that have restriction
function ValueOfBlackCellsNear(k,LightUpTable,NeuronToCell)
    aB = []
    M,N=size(LightUpTable)
    i,j = NeuronToCell[k][2]
    if valid(i-1,j,M,N) && LightUpTable[i-1,j] < BLACK5 push!(aB,LightUpTable[i-1,j]) end
    if valid(i+1,j,M,N) && LightUpTable[i+1,j] < BLACK5 push!(aB,LightUpTable[i+1,j]) end
    if valid(i,j-1,M,N) && LightUpTable[i,j-1] < BLACK5 push!(aB,LightUpTable[i,j-1]) end
    if valid(i,j+1,M,N) && LightUpTable[i,j+1] < BLACK5 push!(aB,LightUpTable[i,j+1]) end
    aB
end

# Return the array of neurons near the black cell i,j that has restriction
function NeuronsNearBlack(i,j,LightUpTable,CellToNeuron)
    aW = []
    M,N=size(LightUpTable)
    if valid(i-1,j,M,N) && LightUpTable[i-1,j] == WHITE push!(aW,CellToNeuron[i-1,j]) end
    if valid(i+1,j,M,N) && LightUpTable[i+1,j] == WHITE push!(aW,CellToNeuron[i+1,j]) end
    if valid(i,j-1,M,N) && LightUpTable[i,j-1] == WHITE push!(aW,CellToNeuron[i,j-1]) end
    if valid(i,j+1,M,N) && LightUpTable[i,j+1] == WHITE push!(aW,CellToNeuron[i,j+1]) end
    aW
end

# Fill the matrix of weights W
# W[i,i] = 0
# W[i,j] = W[j,i] = α if the neuron i shines neuron j
# W[i-1,j] = W[i+1,j] = W[i,j-1] = W[i,j+1] = α*(value of black cell) in (i,j)
function FillWI(LightUpTable,CellToNeuron,NeuronToCell,aC=[α,β,γ])
	M,N,nW,nB = LUConstants(LightUpTable)

	# Objective function properly said
	Q = CreateArray(1,nW,nW)

	# Identity matrix
	I = one(Q)

	# Bulb light bulb constraint
	Qk = CreateArray(0,nW,nW)
    for i in 1:nW, j in i:nW
        Qk[i,j] = Light(i,j,LightUpTable,NeuronToCell)
        Qk[j,i] = Qk[i,j]
    end

	# Tuples of indices of black cells
	aB = []
	for i ∈ 1:M, j ∈ 1:N
		if LightUpTable[i,j] < BLACK5
			push!(aB,(i,j))
		end
	end

	# Blacks constraints
	if len(aB) > 0
		A = CreateArray(0,len(aB),nW)
		b = CreateArray(0,len(aB))
	end
	for (k,ij) ∈ enumerate(aB)
		aN = NeuronsNearBlack(ij...,LightUpTable,CellToNeuron)
		A[k,aN] .= 1
		b[k] = LightUpTable[ij...]
	end

	Aplus = pinv(A)		# Pseudo inverse of A
	P = Aplus*A			   # Projection operator P' = P and P^2=P
	R = I - P
	bhat = Aplus*b
	α = aC[1]; β = aC[2]; γ = aC[3]
	W0 = α*Q - β*Qk
	W = W0 - γ*P
	J = -γ*bhat
    
   DW = diag(W)               # Diagonal of W
   DWM = diagm(DW)            # Matrix only with the W diagonal
   WH = W - DWM               # W with zeros in diagonal. This is the traditional Hopfield W
   JH = J - DW/2              # Hopfield J
   C = γ/2*bhat'*bhat         # This term does not affect the optimization process
   
   W, J, A, Aplus, b, bhat, P, WH, JH, C
end

# Randomly update all neurons of the table
function Hopfield0(X,W=W,J=J,T=Inf,SA=false)
   Xold = copy(X)
	r = collect(1:len(X))
   β = 1/(κ*T)
   while length(r) > 0     # Randomly update all neurons. This is one step
		i, r = randx(r)
		b = (W*X - J)[i]
		X[i] = sgn(b)
		# X[i] = sgn(b,X[i]) # another version of sgn
		ΔX = X[i] - Xold[i]
		ΔE = -W[i,i]*ΔX^2/2 - H(X,W,J)[i]*ΔX
		if SA X[i] = rand() <= exp(-β*ΔE) ? X[i] : Xold[i] end
	end
   dH = sum(xor.(X,Xold))     # Hamming distance of X and Xold
end

# Run Hopfield0 until convergence
function Hopfield(X,W,J,LightUpTable,NeuronToCell,aE,C)
   M,N = size(LightUpTable)
   SA = false
   T = Inf
   ΔE = κ
   ρ = netρ
   HopVal = Inf
	step = 1
	E = -1/2*X'*W*X + J'*X + C
	push!(aE,E)
	Time1 = tic()
   FlagSA = true
	while HopVal !=0
      if step == 1 || step % 1000 == 0 T,SA,ρ=ProcessChangeT(aE,T,SA) end
      if SA && FlagSA 
         Time1 = toc() 
         FlagSA = false
      end
		HopVal = Hopfield0(X,W,J,T,SA)
      E = -1/2*X'*W*X + J'*X + C
		push!(aE,E)
		if (step % STEPSTOPRINT == 0 || step == 1)
         @printf("\nHopfield step %5d, HopVal = %3d, E = %10.4g, ρ = %f, T = %3.2f, Time = %f",
            step, HopVal, E, ρ, T, toc())
      end
      step = step + 1
	end
   Time2 = toc()
   @printf("\nHopfield step %5d, HopVal = %3d, E = %10.4g, ρ = %f, T = %3.2f, , Time = %f\n", 
      step, HopVal, E, ρ, T, Time2)
   if SA 
      @printf("Time in enhanced Hopfiel %f s\n",Time1)
      @printf("Time in sumulated annealing %f s\n",Time2-Time1)
   else
      @printf("Time in enhanced Hopfiel %f s\n",Time2)
   end
   PrintLightUp(X)
   println("\n\n")
   if SA return step, HopVal, Time1, Time2-Time1 end
   step, HopVal, Time2, 0
end

function ProcessChangeT(aE,T,SA)
   ρ,ΔE = fρ(aE)
   if ρ < 0.51 && !SA
      SA = true
      T = 10
      global κ = ΔE
   else
      T = 0.9*T
   end
   T,SA,ρ
end

# Function to calculate the convergence rate
# Input
#	aE:		array with the energy at each step
# Output
#	aρ:		array with the convergence rate after each step
function faρ(aE)
   if len(aE)==1 return netρ,aE[1] end
	aΔE = diff(aE)
	aneg = [(aΔE[k] < 0 ? -aΔE[k] : 0) for k in 1:len(aΔE)]
	aabs = abs.(aΔE)
	aρ = cumsum(aneg)./cumsum(aabs)
	pushfirst!(aρ,netρ)
   aρ, minimum(aΔE), maximum(aΔE)
end

# The same function returning the convergence rate
# of the segment of length len(aE)
function fρ(aE)
   if len(aE)==1 return netρ,aE[1] end
	lena = len(aE)
	start = lena < 1000 ? 1 : lena - 999
	aΔE = diff(aE)
	aneg = [(aΔE[k] < 0 ? -aΔE[k] : 0) for k in 1:len(aΔE)]
	aabs = abs.(aΔE)
   ρ = sum(aneg)/sum(aabs)
   ρ, 2std(aE[start:end])
end

E(X,W,J) = -1/2*X'*W*X + J'*X
H(X,W,J) = W*X - J

# Approximate convergence rate
function ConvergenceRate(W,J)
   N = len(J)
   maxΔE = zeros(N)
   minΔE = zeros(N)
   aρ = zeros(N)
   # Fraction of states given ΔE <= 0 for all neuron i
   for i in 1:N
	   sumNeg = sum(x->x<=0 ? x : 0,W[i,:])
	   sumPos = sum(x->x>0 ? x : 0,W[i,:])
	   H = [abs(sumNeg*s1+sumPos*s2-J[i]) for s1 in [0,1] for s2 in [0,1]]
	   minH = 0
	   maxH = maximum(H)
	   maxΔE[i] = max(0,-W[i,i]/2)
	   minΔE[i] = min(0,-W[i,i]/2 - maxH)
	   aρ[i] = -minΔE[i]/(maxΔE[i]-minΔE[i])
   end
   mean(aρ)
end

# See Leondes pag 238 Neural Network Systems Implemetation Techniques Vol 3
# Any one of the conditions (C1)-(C6) listed below is sufficient
# for the network 𝒩 = (W,I) to converge globally to a stable state when
# operating in the serial mode.
# For all i ∈ 1:N
# C1: W is symmetric and w_{ii} ≥ 0
# C2: w_{ii} ≥ 1/2*∑_{j,j≠i}^N |w_{ij}-w_{ji}|
# C3: w_{ii} ≥ ∑_{j,j≠i}^N |w_{ji}|
# C4: w_{ii} ≥ ∑_{j,j≠i}^N |w_{ij}|
# C5: The matrix B with b_{ii}=W_{ii} and b_{ij}=-|W_{ij}| for i≠j is an M-matrix
# Obs.: According to https://en.wikipedia.org/wiki/M-matrix the matrix B is an
# M-matrix if all their real eigenvalues are positive
function Stable(W)
    nW,nW=size(W)
	if sum(abs.((W-W')/maximum(abs.(W))) .< eps()) == nW^2 && sum(diag(W) .>= 0) == nW
		return 1 	# Condition C1 satisfied
	end

    if sum(diag(W) - sum(abs.(W-W')/2,dims=2) .>= 0) == nW
        return 2    # Condition C2 satisfied
    end

    if sum(diag(W)' - (sum(abs.(W),dims=1)-diag(W)') .>= 0) == nW
        return 3    # Condition C3 satisfied
    end

    if sum(diag(W) - (sum(abs.(W),dims=2)-diag(W)) .>= 0) == nW
        return 4    # Condition C4 satisfied
    end

    B = -abs.(W)
    for k in 1:nW B[k,k]=W[k,k] end
		 λ,U = eigen(B)
    if sum(real(λ) .>= 0) == nW return 5 end   # Condition C5 satisfied

    if ConvergenceRate(W,J) > 0.5 return 6 end
    0
end

# CheckBlack return true if the black constraints are ok
# return false and the indexes of equations that where not satisfied
# The input are the state of the neurons X at the end of iterations
# and the matrices A and b of the equation of constraints
# OUTPUT
#		Boolean:		Indicative that black constraints are satisfied
#		aEqNotOk:	Array of cartesian coordinates (i,j) of black constraints not satisfied
#		EqOk:			Rate of correct black constraint satisfied
function CheckBlack(X,A=A,b=b)
	lenb = len(b)
	r = A*X .== b
	EqNotOk = 0
	aEqNotOk = []
	if sum(r) == lenb return true,[(0,0)],1 end
	ind = findall(x->x==0,r)
	for k in ind
		s=0;
		for i in 1:M, j in 1:N
			if LightUpTable[i,j] < BLACK5
				s = s + 1
				if s == k
					aEqNotOk = push!(aEqNotOk,(i,j))
					break
				end
			end
		end
		EqNotOk = EqNotOk + 1
	end
	EqOk = 1 - EqNotOk/lenb
   false, aEqNotOk, EqOk
end


# Verify if light bulb not iluminate other light bulb
function CheckLightNotSeeLight(X,LightUpTable=LightUpTable,NeuronToCell=NeuronToCell)
	ind = findall(x->x==1,X)
	N = len(ind)
	TotalConds = N*(N-1)/2
	BadConds = 0
	aBadConds = []
	for i in 1:N-1, j in i+1:N
		if Light(ind[i],ind[j],LightUpTable,NeuronToCell) == 1
			l,c = NeuronToCell[ind[i]][2]
			BadConds = BadConds + 1
			push!(aBadConds,(l,c))
		end
	end
	OkConds = 1 - BadConds/TotalConds
	BadConds == 0, aBadConds, OkConds
end

# Verify if all white cells are iluminated
# Return a boolean indicating if all cells are lit: true if r==0
# the list of cartesian coordinates [(i,j)] of the cells not lit
# the rate of cells lit: r/nW
function CheckAllIluminated(X,LightUpTable=LightUpTable,NeuronToCell=NeuronToCell)
	nW = len(X)
	LUT = copy(LightUpTable)
	L,C = size(LUT)
	ind = findall(x->x==1,X)	# Find all bulbs on the table
	# Put bulbs in all places that is lit
	for k in 1:len(ind)
		l,c = NeuronToCell[ind[k]][2]
		LUT[l,c] = BULB
		for k in l:-1:1
			if (LUT[k,c] == WHITE) || LUT[k,c] == BULB
				LUT[k,c] = BULB
			else
				break
			end
		end
		for k in l:L
			if (LUT[k,c] == WHITE) || LUT[k,c] == BULB
				LUT[k,c] = BULB
			else
				break
			end
		end
		for k in c:-1:1
			if (LUT[l,k] == WHITE) || LUT[l,k] == BULB
				LUT[l,k] = BULB
			else
				break
			end
		end
		for k in c:C
			if (LUT[l,k] == WHITE) || LUT[l,k] == BULB
				LUT[l,k] = BULB
			else
				break
			end
		end
	end
	WhitesNotLit = 0
	aWhitesNotLit = []
	for i in 1:L, j in 1:C
		if LUT[i,j] == WHITE
			WhitesNotLit = WhitesNotLit + 1
			push!(aWhitesNotLit,(i,j))
		end
	end
	r = 1 - WhitesNotLit/nW
	WhitesNotLit == 0, aWhitesNotLit, r
end

# Generate a lightup table with M lines and N columns
# with B% of black cells
# This routine generate an inconsistent lightup table (fix later)
function GenLU(M,N,B)
   LU = 6Int.(ones(M*N))      # fill lightup with white cells
   for k in 1:M*N
      if rand() <= B/100    # black
         b = rand([0,1,2,3,4,5])
         LU[k] = b
      end
   end
   LU = reshape(LU,M,N)
   for i in [1,M], j in [1,N]
      if 2 < LU[i,j] < 5 LU[i,j] = 2 end
   end
   for i in [1,M], j in [2:N-1]
      if LU[i,j] == 4 LU[i,j] = 3 end
   end
   for i in [2,M-1], j in [1:N]
      if LU[i,j] == 4 LU[i,j] = 3 end
   end
   for i in 1:M, j in 1:N
      if LU[i,j] < 6
         nw = 0
         for l in [i-1,i+1] if valid(l,j,M,N) && LU[l,j] == 6 nw = nw + 1 end end
         for c in [j-1,j+1] if valid(i,c,M,N) && LU[i,c] == 6 nw = nw + 1 end end
         if LU[i,j] > nw LU[i,j] = nw end
      end
   end
   LU
end

function SaveLU(txtfilename,LUTable)
   open(txtfilename,"w") do io
      writedlm(io,LUTable)
   end
end


#
# Main program
#

# Read light up puzzle
print("Enter file to process: (Ex.: luem03.txt) ")
strfile = readline()
println("Processing file: $strfile")

LightUpTable = Int.(readdlm(strfile))

# Generate LU table (need to fix to obtain a valid table)
# LightUpTable = GenLU(25,25,10)

# Calculate light up constants
M, N, nW, nB = LUConstants(LightUpTable)


# Adjustable parameters of the energy function
α = 4nW		 	# A convenient parameter multiplying the objective function
β = 800nW		# Lagrange multiplyier for bulbs on the same line or on the same column
γ = 10000nW    # Lagrange multiplyier for black constraints terms
κ = Inf 			# Boltzman constant
aE = []			# Array of energies in each step

# Create the neuron X array,
# the array of correspondence between neuron and cells, NeuronToCell
# and the correspondence matriz between cells and neurons, CellToNeuron.
# When X[j] == 1 there is a bulb in j.

X, NeuronToCell, CellToNeuron = Neuron(LightUpTable)
W, J, A, Aplus, b, bhat, P, WH, JH, C = FillWI(LightUpTable,CellToNeuron,NeuronToCell,[α,β,γ])

netρ = ConvergenceRate(W,J)      

if Stable(W) == 0
	println(Crayon(foreground=:yellow,background=:red),"\n\nWARNING: Network may not be stable.")
	println(Crayon(reset=true))
end


# Print initial configuration
println(Crayon(foreground=:white,background=:blue),"\n\nInitial configuration")
PrintLightUp(X,LightUpTable,NeuronToCell)

# Run the program
Steps, HopVal, THF, TSA = Hopfield(X,W,J,LightUpTable,NeuronToCell,aE,C)

# Run traditional Hopfield if the program ended with Hamming distance HopVal > 0
if HopVal > 0
	println("Entering traditional Hopfield mode")
	STEPSTOPRINT = 1
	Steps, HopVal, TTHF, TTSA = Hopfield(X,WH,JH,LightUpTable,NeuronToCell,aE,C)
end

# Verify the solution
SolBlack, aSolBlack, rSolBlack = CheckBlack(X)
SolLight, aSolLight, rSolLight = CheckLightNotSeeLight(X)
SolAll, aSolAll, rSolAll = CheckAllIluminated(X)
HitRate =100(rSolBlack+rSolLight+rSolAll)/3

if SolBlack
	println(Crayon(background=:green,foreground=:white),"\n\n Black constraints SOLVED ")
	println(Crayon(reset=true))
else
	println(Crayon(background=:red,foreground=:white),"\n\n Black constraints not SOLVED ")
	println(Crayon(background=:red,foreground=:white),
		"\n\n Black cells $(aSolBlack) not satisfied")
	println(Crayon(reset=true))
end

if SolLight
	println(Crayon(background=:green,foreground=:white),"\n\n Light not see light constraints SOLVED ")
	println(Crayon(reset=true))
else
	println(Crayon(background=:red,foreground=:white),
		"\n\n Light in $(aSolLight) see other NOT SOLVED ")
	println(Crayon(reset=true))
end

if SolAll
	println(Crayon(background=:green,foreground=:white),"\n\n All white cells iluminated SOLVED")
	println(Crayon(reset=true))
else
	println(Crayon(background=:red,foreground=:white),
		"\n\n Cell $(aSolAll) not iluminated: NOT SOLVED")
	println(Crayon(reset=true))
end

aρ,ΔEmin,ΔEmax = faρ(aE)

# Print results
@printf("\nNetwork Convergence Rate: %3.2f",netρ);
@printf("\nConvergence rate after %d steps: %3.2f",Steps,aρ[end])
@printf("\nHit rate after %d steps: %5.2f",Steps,HitRate);print("%")
@printf("\nBoltzman constante = %5g\n\n",κ)

# Plot the energy function and convergence rate
close("all")
aIter = collect(0:len(aE)-1)
figure(1)
	subplot(211)
   	plot(aIter,aE/maximum(abs.(aE)))
		title("Objective Function and Convergence Rate")
		ylabel("E")
		minorticks_on()
		grid(b=true, which="major", color="#666666", linestyle="-")
		grid(b=true, which="minor", color="#999999", linestyle="-", alpha=0.2)

   	subplot(212)
   		plot(aIter,aρ)
   		ylabel(L"\rho")
   		xlabel("Hopfield step")
		minorticks_on()
		grid(b=true, which="major", color="#666666", linestyle="-")
		grid(b=true, which="minor", color="#999999", linestyle="-", alpha=0.2)


# Generate LaTeX code for lightup puzzle given the configuration X 
function TeX(X,sfile="lutex.out")
	file = open(sfile,"w")
   LUT = copy(LightUpTable)
   L, C = size(LUT)
   ind = findall(x->x==1,X)	# Find all bulbs on the table
	for k in 1:len(ind)
		l,c = NeuronToCell[ind[k]][2]
		LUT[l,c] = BULB
   end
   i3 = "   "
   i6 = i3*i3
   println(file,"\\begin{center}")
   println(file,"$i3\\scalebox{0.55}{")
   print(file,"$i3\\begin{tabular}{|")
   for n in 1:C print(file,"c|") end; println(file,"}")
   for i in 1:L
      println(file,"$i6\\hline")
      print(file,i6)
      for j in 1:C
         if j < C strend = " & " else strend = "\\\\\n" end
         if LUT[i,j] < BLACK5 
            print(file,"\\cellcolor{black}\\color{white}{\\bf $(LUT[i,j])}$strend") 
         elseif LUT[i,j] == BLACK5
            print(file,"\\cellcolor{black}$strend") 
         elseif LUT[i,j] == BULB
            print(file,"\\faLightbulbO$strend") 
         else  # WHITE
            print(file,"\\phantom{0}$strend") 
         end
      end
      if i == L println(file,"$i6\\hline") end
   end
   println(file,"$i3\\end{tabular}}")
   println(file,"\\end{center}")
   close(file)
end

TeX(X,"HP.out")

# Do means for statistics
if DOMEANS
   close("all")
	println("Calculations for statistics")

	aTHF = []
	aTSA = []
	aHR = []
	aSteps = []
	aρend = []
	
	for k in 1:30
		local aE = []
		local κ = Inf 
		local X, NeuronToCell, CellToNeuron
		local Steps, HopVal, THF, TSA
		local SolBlack, aSolBlack, rSolBlack
		local SolLight, aSolLight, rSolLight
		local SolAll, aSolAll, rSolAll

		X, NeuronToCell, CellToNeuron = Neuron(LightUpTable)
		Steps, HopVal, THF, TSA = Hopfield(X,W,J,LightUpTable,NeuronToCell,aE,C)
		SolBlack, aSolBlack, rSolBlack = CheckBlack(X)
		SolLight, aSolLight, rSolLight = CheckLightNotSeeLight(X)
		SolAll, aSolAll, rSolAll = CheckAllIluminated(X)
		local HitRate =100(rSolBlack+rSolLight+rSolAll)/3
		local aρ,ΔEmin,ΔEmax = faρ(aE)
		push!(aTHF,THF)
		push!(aTSA,TSA)
		push!(aHR,HitRate)
		push!(aSteps,Steps)
		push!(aρend,aρ[end])
	end

	mTHF = mean(aTHF)
	σTHF = std(aTHF)

	mTSA = mean(aTSA)
	σTSA = std(aTSA)

	mHR = mean(aHR)
	σHR = std(aHR)

	mSteps = mean(aSteps)
	σSteps = std(aSteps)

	mρend = mean(aρend)
	σρend = std(aρend)

	println(Crayon(background=:yellow,foreground=:black))
	println("N: $nW")
	println("netρ: $netρ")
	println("ρend: $mρend")
	println("Steps: $mSteps")
	println("Hit rate: $mHR ± $σHR")
	println("Hopfiel mode time: $mTHF ± $σTHF")
	println("Simulated annealing mode time: $mTSA ± $σTSA")
end
