
using JuMP, GLPK
function check_feasible(LR_x,  facility_data, customers, a, b)
    is_feasible = true
    violated_facility=[]
    for i in 1 : length(facility_data)
        if sum(a[j]*LR_x[i,j] for j in 1:length(customers)) > b[i] # capacity 제약조건 (1) 문제 성립여부 확인
            push!(violated_facility, i)
            is_feasible = false
        end
    end
    return is_feasible, violated_facility
end

function lambda_update(lambda, diff_previous, w, Z_UB, Z_LB, a, LR_x, b, U, customers)
    lambda_decrease = []
    diff_current = zeros(length(lambda))

    for i in U
        numerator = sum(a[j] * LR_x[i, j] for j in 1:length(customers)) - b[i] 
        denominator = sqrt(sum((sum(a[j] * LR_x[k, j] for j in 1:length(customers)) - b[i])^2 for k in U))

        new_lambda = max(0, lambda[i] + (w * (Z_UB - Z_LB) * numerator / denominator))
    
        diff_current[i] = new_lambda - lambda[i]
        
        if diff_previous[i] < 0 && diff_current[i] > 0
            push!(lambda_decrease, i)
        end

        lambda[i] = new_lambda
    end
    U = setdiff(U, lambda_decrease)

    return lambda, diff_current, U
end

function optimal(facility_data, customers, a, b, f, c)
    m = Model(GLPK.Optimizer)
    @variable(m, x[i in 1:length(facility_data), j in 1:length(customers)], Bin)
    @variable(m, y[i in 1:length(facility_data)], Bin)

    @objective(m, Min, sum(f[i] * y[i] for i in 1:length(facility_data)) + sum(c[i,j] * x[i,j] for i in 1:length(facility_data), j in 1:length(customers)))

    @constraint(m, [i in 1:length(facility_data)], sum(a[j]*x[i,j] for j in 1:length(customers)) <= b[i] )
    @constraint(m, [j in 1:length(customers)], sum(x[i,j] for i in 1:length(facility_data)) == 1)
    @constraint(m, [i in 1:length(facility_data), j in 1:length(customers)], x[i,j] <= y[i])

    JuMP.optimize!(m)

    Z_opt = JuMP.objective_value(m)
    x_opt = JuMP.value.(x)
    y_opt = JuMP.value.(y)

    return Z_opt, x_opt, y_opt
end
# Z_opt, x_opt, y_opt = optimal(facility_data, customers, a, b, f, c)
function Lagrangian_Relaxation(facility_data, customers, a, b, f, c, lambda)
    m = Model(GLPK.Optimizer)
    @variable(m, x[i in 1:length(facility_data), j in 1:length(customers)], Bin)
    @variable(m, y[i in 1:length(facility_data)], Bin)

    @objective(m, Min, sum(f[i] * y[i] for i in 1:length(facility_data))
                        + sum((c[i,j]+ lambda[i]*a[j]) * x[i,j] for i in 1:length(facility_data), j in 1:length(customers)) 
                        - sum(lambda[i]*b[i] for i in 1:length(facility_data))
                        )

    @constraint(m, [j in 1:length(customers)], sum(x[i,j] for i in 1:length(facility_data)) == 1)
    @constraint(m, [i in 1:length(facility_data), j in 1:length(customers)], x[i,j] <= y[i])

    JuMP.optimize!(m)

    Z_opt = JuMP.objective_value(m)
    x_opt = JuMP.value.(x)
    y_opt = JuMP.value.(y)

    return Z_opt, x_opt, y_opt
end

function generate_customers(n)
    customers = []
    demands = []
    for i in 1:n
        index = i
        longitude = rand() * 180 - 90  # -90 to 90
        latitude = rand() * 360 - 180  # -180 to 180
        demand = rand() * 100000 + 100000 # 0 to 50000
        push!(customers, (index, longitude, latitude, demand))
        push!(demands, demand)
    end

    return customers, demands
end


function data_setting(lines)
    # 결과를 저장할 배열 초기화
    extracted_data = []
    capacities = []
    fixed_costs = []
    # 각 줄을 순회하면서 필요한 데이터 추출
    for line in lines
        # 공백을 기준으로 줄을 나누고 필요한 부분만 선택
        parts = split(line)
        index = parse(Int, parts[1]) 
        longitude = parse(Float64, parts[2])  
        latitude = parse(Float64, parts[3])
        capacity = parse(Int, parts[5])
        fixed_cost = parse(Int, parts[6])
        push!(extracted_data, (index, longitude, latitude, capacity, fixed_cost))
        push!(capacities, capacity)
        push!(fixed_costs, fixed_cost)
    end
    
    return extracted_data, capacities, fixed_costs
end


function cost_matrix(facility_data, customers)

    matrix = zeros(length(facility_data), length(customers))
    for i in 1 : length(facility_data)
        for j in 1 : length(customers) 
            matrix[i,j] = distance_haversine(facility_data[i][2], facility_data[i][3], customers[j][2], customers[j][3])
        end
            # println(extracted_data[i][j])

    end
    return matrix
end

function distance_haversine(lon1, lat1, lon2, lat2)
    R = 6371
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    d = R * c
    return d
end



function optimize_process(facility_data, customers, a, b, c, f)
    K = []  # 초기에 열린 시설
    not_in_K = setdiff(1:length(facility_data), K)
    dont_reopen = []
    Z_UB = Inf
    is_cycle = true

    while is_cycle
        # Step 1 - Calculate R
        
        not_in_K = setdiff(not_in_K, dont_reopen) # dont_reopen은 재오픈 안할 시설 

        R = calculate_R(not_in_K, K, facility_data, customers, a, b, c, f)

        # Step 2 - Open new facility
        new_facility = argmax(R)
        if R[new_facility] == -Inf
            break  # 종료 조건
        end
        push!(K, new_facility)
        not_in_K = setdiff(not_in_K, K)

        # Step 3 - Check capacity
        if sum(b[i] for i in K) < sum(a[j] for j in 1:length(customers))
            continue  # 처음으로 돌아가기
        end

        # Step 4 
        cost_diff = zeros(length(customers))
        for j in 1:length(customers)
            if !isempty(K)
                sorted_c = sort(collect(c[i,j] for i in K))
                print(sorted_c)
                min_value = sorted_c[1]
                if length(K) > 1
                    second_min_value = sorted_c[2]
                    cost_diff[j] = second_min_value - min_value
                end
            end
        end

        if length(K) == 1
            cost_diff = []
            for j in 1:length(customers)
                append!(cost_diff, c[K[1], j])
            end
            sorted_indices = sortperm(cost_diff) # K 한 개 일때, k facility(단일) 까지의 거리 순서로 오름차순 정렬
        else
            sorted_indices = sortperm(cost_diff, rev=true) # 큰 값부터 정렬
        end


        # Step 5 --------------------------------------------
        remaining_capacity = collect(b[i] for i in 1:length(facility_data))
        assignment = zeros(Int64, length(customers))
        is_assigned = []

        for j in sorted_indices
            possible_assignment = []
            for i in K
                if a[j] <= remaining_capacity[i]
                    push!(possible_assignment, i)
                end
            end

            if !isempty(possible_assignment)
                min_cij = Inf
                for i in possible_assignment
                    if min_cij > c[i,j]
                        min_cij = c[i,j]
                        assignment[j] = i
                    end
                end
                # assignment[j] = argmin([c[i,j] for i in possible_assignment])
                remaining_capacity[assignment[j]] -= a[j]

            else # 가능한 facility 없을 경우
                println("go to step1 - no possible facility!")
                continue
                # go to step 1
            end
            
        end

        
        # Step 6 - Finalization
        is_assigned = unique(assignment)
        if length(is_assigned) != length(K)
            not_assigned = setdiff(K, is_assigned)
            push!(dont_reopen, not_assigned)
            K = setdiff(K, not_assigned)
            continue  # 처음으로 돌아가기
        end


        x = zeros(length(facility_data), length(customers))

        for j in 1:length(customers)
            num = assignment[j]
            if num != 0 
                x[num,j] = 1
            end
        end

        y = zeros(length(facility_data))
        for i in K
            y[i] = 1
        end

        Total_Cost = sum(f[i] * y[i] for i in K) + sum(c[i,j] * x[i,j] for i in K, j in 1:length(customers))


        if Total_Cost < Z_UB
            Z_UB = Total_Cost
            continue  # Update and continue
        else
            is_cycle = false  # 종료
        end
    end

    return Z_UB
end


function calculate_R(not_in_K, K, facility_data, customers, a, b, c, f)
    w = zeros(length(facility_data), length(customers))
    Ω = zeros(length(facility_data))
    R = fill(-Inf, length(facility_data))
    for i in not_in_K
        for j in 1:length(customers)
            if length(K) == 0
                w[i,j] = max(min(0 - c[i,j]) , 0) 
                # println(w[i,j])
            else
                w[i,j] = max(minimum(collect(c[k,j] - c[i,j] for k in K)), 0)
                # println(w[i,j])
            end
        end
    end

    for i in not_in_K
        Ω[i] = sum(w[i,j] for j in 1:length(customers))
    end
    possible_j = []
    for i in not_in_K
        for j in 1:length(customers)
            
            if w[i,j] > 0
                push!(possible_j, j)
            end
        end

        if length(possible_j) != 0
            divider = sum(a[j] for j in possible_j)
            R[i] = Ω[i] * min(b[i] / divider,1) - f[i]
        else
            R[i] = Ω[i] - f[i] # sum a[j] for j in possible_j (공집합일 때) 
        end
    end
    return R
end