using JuMP, GLPK

# read the file
file = open("sortcap.grt", "r")
lines = readlines(file)
close(file)
# facility data 로드
# index(i), longitude, latitude, capacity, fixed_cost
facility_data, b, f = data_setting(lines)
# customer data 생성
# index(i), longitude, latitude, demand
customers, a = generate_customers(20)

# cost 생성
c = cost_matrix(facility_data, customers)


# ---------------------------------------------------------------------
# another data
facility_data = collect(1:7)
customers = collect(1:10)

a = [10, 6, 20, 32, 15, 28, 3, 19, 8, 13]
c = [
    [10, 7, 11, 12, 32, 15, 20, 26, 4, 41],
    [13, 17, 31, 37, 21, 5, 13, 15, 14, 12],
    [4, 13, 14, 22, 8, 31, 26, 11, 12, 23],
    [21, 21, 13, 18, 9, 27, 11, 16, 26, 32],
    [32, 18, 11, 14, 11, 11, 16, 32, 34, 8],
    [15, 9, 13, 12, 14, 15, 32, 8, 12, 9],
    [28, 32, 15, 2, 17, 12, 9, 6, 11, 6]
]

# Vector{Vector{Int}}에서 Matrix{Float64}로 변환
c = transpose(Float64.(reduce(hcat, map(vec -> reshape(vec, :, 1), c))))
b = [50, 50, 50, 50, 50, 100, 100]
f = [10, 10, 10, 10, 10, 30, 30]

# b = [50, 50, 50, 50, 50, 100, 100]
# f = [30, 30, 30, 30, 30, 10, 10]

## Lagrangian relaxation heuristic
MAX_iter = 200
iteration = 0
feasible_sol = Inf
is_finished = false

# Step 0 ----------------------------------------------
Z_UB = 150 # add heuristic 사용
lambda = zeros(length(facility_data))
LR_opt, LR_x, LR_y = Lagrangian_Relaxation(facility_data, customers, a, b, f, c, lambda)
Z_LB = LR_opt


is_feasible, violated_facility = check_feasible(LR_x, facility_data, customers, a, b)

U = Vector{Int64}()
if is_feasible == false
    # violated_facility의 lambda initialize
    for i in violated_facility
        push!(U, i)
        U = unique(U)
    end
    diff_previous = zeros(length(lambda))
    lambda, diff_current, U = lambda_update(lambda, diff_previous, Z_UB, Z_LB, a, LR_x, b, U, customers)
end
U = collect(1: length(facility_data)) # mark 초기화


while is_feasible == false    
    # ------------------------------------------------
    # Step 1
    LR_opt, LR_x, LR_y = Lagrangian_Relaxation(facility_data, customers, a, b, f, c, lambda)
    if LR_opt > Z_LB
        Z_LB = LR_opt
    end
    # feasibility self-check_and_resize
    
    # current_LR_cost = sum(f[i] * LR_y[i] for i in 1:length(facility_data)) + sum((c[i,j]+ lambda[i]*a[j]) * LR_x[i,j] for i in 1:length(facility_data), j in 1:length(customers)) - sum(lambda[i]*b[i] for i in 1:length(facility_data))
    # + sum((c[i,j]+ lambda[i]*a[j]) * LR_x[i,j] for i in 1:length(facility_data), j in 1:length(customers)) 
    # - sum(lambda[i]*b[i] for i in 1:length(facility_data))

    # println("current_LR_cost: ", current_LR_cost)

    is_feasible, violated_facility = check_feasible(LR_x,  facility_data, customers, a, b)
    iteration += 1
    println("--------------------------------------------------------------")
    println("iteration: ", iteration, " Z_LB: ", Z_LB, " Z_UB: ", Z_UB, " is_feasible: ", is_feasible, " violated_facility: ", violated_facility)
    println("LR_opt: ", LR_opt)

    if is_feasible == true
        println("find feasible solution!")
        objective_val = sum(f[i] * LR_y[i] for i in 1:length(facility_data)) + sum(c[i,j] * LR_x[i,j] for i in 1:length(facility_data), j in 1:length(customers))
        println("objective_val: ", objective_val)
        if objective_val < Z_UB
            Z_UB = objective_val
        else
            is_finished = true
            println("a better solution was already found.")
            break
        end


        # go to step 4
        ## step 4 ------------------------------
        if Z_UB/Z_LB <=  1 + 1e-6
            is_finished = true
            println("optimality gap is less than 1 + 1e-6.")
            break

        else
            # go to step 3
            U = collect(1: length(facility_data)) # mark 초기화
        end
        ## ------------------------------------
    end

    # Step 2
    if iteration > MAX_iter
        is_finished = true
        break
        # go to step 5 
    end

    # Step 3
    diff_previous = diff_current
    for i in violated_facility
        push!(U, i)
        U = unique(U)
    end
    lambda, diff_current, U = lambda_update(lambda, diff_previous, Z_UB, Z_LB, a, LR_x, b, U, customers)
    is_feasible = false # 다시 1로 돌아감
    # go to step 1
end

#Step 5

if is_finished == true
    println("Z_UB: ", Z_UB)
    println("Z_LB: ", Z_LB)
end

Z_opt, x_opt, y_opt = optimal(facility_data, customers, a, b, f, c)
println("optimal solution of solver: ", Z_opt)







