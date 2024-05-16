## data loading
using JuMP, GLPK
include("functions.jl")
file = open("sortcap.grt", "r")
lines = readlines(file)
close(file)

facility_data, b, f = data_setting(lines)
f = f ./10
b = b .* 2
customers, a = generate_customers(20)
a = round.(Int64,a)
c = cost_matrix(facility_data, customers)
Z_UB = optimize_process(facility_data, customers, a, b, c, f)



## Initial add heuristic

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