using JuMP, GLPK, Random
include("functions.jl")

file = open("sortcap.grt", "r")
lines = readlines(file)
close(file)
# facility data 로드
# index(i), longitude, latitude, capacity, fixed_cost
facility_data, b, f = data_setting(lines)
f = f ./10
b = b .* 2

Random.seed!(500) # 500 - 28 iter
customers, a = generate_customers(20)
a = round.(Int64,a)
c = cost_matrix(facility_data, customers)




## Lagrangian relaxation heuristic
MAX_iter = 200
iteration = 0
is_finished = false
feasible_sol = Inf

# Step 0 ----------------------------------------------
# Z_UB = 250000 
Z_UB = optimize_process(facility_data, customers, a, b, c, f) # add heuristic 사용
lambda = zeros(length(facility_data))
LR_opt, LR_x, LR_y = Lagrangian_Relaxation(facility_data, customers, a, b, f, c, lambda)
Z_LB = LR_opt
w = 0.001 # 라그랑주 승수 업데이트 step_size

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
            # 이전에 찾은 해가 더 좋은 (1)의 solution인 경우, 바로 종료
            is_finished = true
            println("a better solution was already found.")
            break
        end
       

        feasible_sol = objective_val # 이전에 찾은 해 저장

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
        println("no feasible sol!")
        break
        # go to step 5 
    end

    # Step 3
    diff_previous = diff_current
    for i in violated_facility
        push!(U, i)
        U = unique(U)
    end
    lambda, diff_current, U = lambda_update(lambda, diff_previous, w, Z_UB, Z_LB, a, LR_x, b, U, customers)
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





