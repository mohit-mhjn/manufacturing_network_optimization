## import required libraries
import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)
import pandas
from pyomo.environ import *
from solutionmethod import solve_model
from pprint import pprint
import itertools

## Problem Specs #####
p_serv_level = 80   ## Percent
km_serv_dist = 500  ## miles
product_plant_assignment = {1:{1:1,2:0,3:0,4:0,5:0},     ## Binary Indicator
                            2:{1:0,2:1,3:0,4:0,5:0},
                            3:{1:0,2:0,3:1,4:0,5:0},
                            4:{1:0,2:0,3:0,4:1,5:1}}
std_truck_capacity = 10 ## Tonnes
cost_per_truck_per_mile = 2   ## $
setup_cost_per_day = 5000 ## $

production_rate = {
                    1:100,    ## Tonnes per hour
                    2:50,
                    3:50,
                    4:50
                }

plant_upgrade_cost = 10000000 ## $
w_hours = 8 ## Per day
overtime = 120  ## Hours per month
cost_factor_overtime = 50 ## Percent

## Read input files
dataset_plants = pandas.read_excel("data.xlsx",sheet_name="Plants")
dataset_customers = pandas.read_excel("data.xlsx",sheet_name="Customers")
dataset_products = pandas.read_excel("data.xlsx",sheet_name="Product")
dataset_demand = pandas.read_excel("data.xlsx",sheet_name="Annual Demand")
dataset_capacity = pandas.read_excel("data.xlsx",sheet_name="Production Capacity")
dataset_distances = pandas.read_excel("data.xlsx",sheet_name="Distances")
dataset_setup = pandas.read_excel("data.xlsx",sheet_name="Setups")

## Transform distance matrix to kv pairs
dataset_distances_1 = dataset_distances.filter(items = ["Plant Id","Customer ID","Distance"])
dataset_distances_1 = dataset_distances_1.dropna()
plant_to_customer = dataset_distances_1.set_index(["Plant Id","Customer ID"]).to_dict(orient = "index")
dataset_distances_2 = dataset_distances.filter(items = ["Customer ID.1", "Customer ID.2","Distance.1"])
customer_to_customer = dataset_distances_2.set_index(["Customer ID.1","Customer ID.2"]).to_dict(orient = "index")
capacity_cost = dataset_capacity.set_index(["Plant ID","Product ID"]).to_dict(orient="index")

dataset_setup.reset_index(drop = False,inplace = True)
dataset_setup = pandas.melt(dataset_setup, id_vars=["index"], value_vars = ['Clear','Green','Red','Blue','Gray'])
dataset_setup = dataset_setup.merge(dataset_products, how = "left", left_on = "index", right_on = "Name")
dataset_setup = dataset_setup.merge(dataset_products, how = "left", left_on = "variable", right_on = "Name")
dataset_setup.rename(index =str, columns = {"ID_x":"from","ID_y":"to","value":"days"},inplace = True)
dataset_setup.drop(labels = ['index','variable','Name_x','Name_y'],axis = 1,inplace= True)
changeover_mapping = dataset_setup.set_index(["from","to"]).to_dict(orient = "index")


## initialize model object
print ("Inputs processed successfully \nInitializing MILP model")
model = ConcreteModel()

## Define indexes
model.plants = Set(initialize = list(dataset_plants["ID"]))  # Set of Plants
model.customers = Set(initialize = list(dataset_customers["ID"])) # Set of customers
model.products = Set(initialize = list(dataset_products["ID"]))    # Set of Products
model.potential_WH = Set(initialize = list(dataset_customers["ID"]))  # Potential WH locations (all customer locations)
# model.dummy_product = Set(initialize = [0])

# ## Product vs Plant combinations
# model.plant_product = Set(dimen = 2, initialize = ((pl,pr) for pr in model.products for pl in model.plants if product_plant_assignment[pl][pr] == 1))

print ("Indexes :: Completed")
## Define parameters
model.BigM = Param(initialize = 10**6)    ## For indicator variable constraints

## Annual Customer Demand >> Quarterly Demand
demand_revenue = dataset_demand.set_index(["Customer ID","Product ID"]).to_dict(orient="index")
def demand_generator(model,c,p):
    global demand_revenue
    return demand_revenue[(c,p)]['Demand (in tonnes)']/4
model.demand = Param(model.customers, model.products, initialize = demand_generator)

def revenue_generator(model,c,p):                               # Revenue per ton
    global demand_revenue
    return demand_revenue[(c,p)]['Revenue ($)']
model.revenue = Param(model.customers, model.products, initialize = revenue_generator)

## Annual capactiy to Quarterly Capacity
def capacity_generator(model,pl,pr):                        ## Capacity of plant by product
    global capacity_cost
    return capacity_cost[(pl,pr)]["Annual Production Capacity"]/4
model.plant_capacity = Param(model.plants, model.products, initialize = capacity_generator)  ## Capacity per quarter

def production_cost_generator(model,pl,pr):    ## Production Cost per ton
    global capacity_cost
    return round(capacity_cost[(pl,pr)]["Production Cost"],1)
model.production_cost = Param(model.plants, model.products, initialize = production_cost_generator)

def production_rate_gen(model,pl):      ## Throughput rate of the plants
    global production_rate
    return production_rate[pl]
model.rate_of_production = Param(model.plants, initialize = production_rate_gen)

def changeover_gen(model,f,t):       ## Seq dependent changeover time
    global changeover_mapping
    return changeover_mapping[(f,t)]["days"]
model.changeover_days = Param(model.products, model.products, initialize = changeover_gen)

## Revenue Earned per quarter
## Demand in ton * Revenue per ton
## Assumption :: the demand is considered consistant, given no other customers are onboarded
def summation_revenue(model):
    return sum(model.demand[c,p]*model.revenue[c,p] for c in model.customers for p in model.products)
model.total_revenue = Param(initialize = summation_revenue)

print ("Params :: Completed")
## Define decision variables
model.x_plant_to_WH = Var(model.plants, model.products, model.potential_WH, within = NonNegativeReals)       ## Quantity of product p shipped from plant pl to warehouse w
model.x_WH_to_customer = Var(model.potential_WH, model.products, model.customers, within = NonNegativeReals) ## Quantity of product p shipped from warehouse w to customer c
model.x_plant_to_customer = Var(model.plants, model.products, model.customers, within = NonNegativeReals)    ## Quantity of product p shipped from plant p to customer c

model.y_plant_to_WH = Var(model.plants, model.products, model.potential_WH, within = Binary)
## Indicator variable that sugessest if any quantity of product p is shipped to warehouse w from plant pl

model.Q = Var(model.plants, model.products, within = NonNegativeReals)     ## Quantity(in tonnes) of product pr produced at plant p
model.yp = Var(model.plants, model.products, within = Binary)              ## Indicator:: Is product pr produced at plant p ?
model.Cp = Var(model.plants, model.products, model.products, within = Binary)  ## Assignment matrix of a plant changeover from product p1 to product p2

## Model Constraints

## Indicator Variable relationships
def x_y_relation1(model,src,p,dst):
    return model.y_plant_to_WH[src,p,dst]*model.BigM >= model.x_plant_to_WH[src,p,dst]
model.warehouse_indicator = Constraint(model.plants, model.products, model.potential_WH, rule = x_y_relation1)

def x_y_relation2(model,pl,pr):
    return model.yp[pl,pr]*model.BigM >= model.Q[pl,pr]
model.production_indicator = Constraint(model.plants, model.products, rule = x_y_relation2)

def y_y_relation1(model,pl,pr):
    return model.yp[pl,pr] == sum(model.Cp[pl,pr,p] for p in model.products)
model.changeover_mtx1 = Constraint(model.plants, model.products, rule = y_y_relation1)

def y_y_relation2(model,pl,pr):
    return model.yp[pl,pr] == sum(model.Cp[pl,p,pr] for p in model.products)
model.changeover_mtx2 = Constraint(model.plants, model.products, rule = y_y_relation2)

def y_y_relation3(model,pl):
    return sum(model.Cp[pl,pr,pr] for pr in model.products) <= 1
model.changeover_mtx3 = Constraint(model.plants, rule = y_y_relation3)

def y_y_relation4(model,pl):
    sum_of_entries = sum(model.Cp[pl,p1,p2] for p1 in model.products for p2 in model.products)
    return sum_of_entries == sum(model.yp[pl,p] for p in model.products)
model.changeover_mtx4 = Constraint(model.plants, rule = y_y_relation4)

##### Following are the flow conservation constraints

## Type 1 node is plant :: total outflow (to customers and warehouses) <= Production at plant
def flow_consv1(model,pl,pr):
    production = model.Q[pl,pr]
    outflow = sum(model.x_plant_to_WH[pl,pr,w] for w in model.potential_WH) + sum(model.x_plant_to_customer[pl,pr,c] for c in model.customers)
    return outflow == production
model.production_flow = Constraint(model.plants, model.products,rule = flow_consv1)

## Type 2 node is warehouse :: total outflow (to customers) <= total inflow (from plants)
def flow_consv2(model,p,w):
    inflow = sum(model.x_plant_to_WH[src,p,w] for src in model.plants)
    outflow = sum(model.x_WH_to_customer[w,p,dst] for dst in model.customers)
    return outflow == inflow
model.warehouse_flow_conserve = Constraint(model.products, model.potential_WH, rule = flow_consv2)

# Type 3 node is Customer :: total inflow (from plant + warehouse) >= demand
def satisfaction_eqn(model,c,p):
    return sum(model.x_WH_to_customer[src,p,c] for src in model.potential_WH) + sum(model.x_plant_to_customer[src,p,c] for src in model.plants) == model.demand[c,p]
model.demand_satisfaction = Constraint(model.customers, model.products, rule = satisfaction_eqn)

## Following are the capacity constraints

## Capacity in measured in terms of the running hours available in a quarter
## total running hours <= max_available hours (regular + overtime)
def capacity_of_plant1(model,pl):
    total_capacity_available = 3*(30*w_hours + overtime) ## Hours of Production  >> eg: (3 months/q x (30 day/month x 8 hrs/day + 120 hrs/month)) == hrs/q
    running_hours = sum(model.Q[pl,pr]/model.rate_of_production[pl] for pr in model.products)  ## (tonnes/q) divided by (tonnes/hrs) == hrs/q
    changeover_hours = sum(w_hours*(model.Cp[pl,p1,p2]*model.changeover_days[p1,p2]) for p1 in model.products for p2 in model.products)   ## changeover_days * 8 hrs a day == changeover time
    capacity_utilized = running_hours + changeover_hours
    return capacity_utilized <= total_capacity_available
model.limit_production_capacity = Constraint(model.plants, rule = capacity_of_plant1)

## Capacity on Individual plant v/s product combinations
## Production Quanity <= specified quarterly capacity
def capacity_of_plant2(model,pl,pr):
    return model.Q[pl,pr] <= model.plant_capacity[pl,pr]
model.limit_product_capacity = Constraint(model.plants, model.products,rule = capacity_of_plant2)

### Cost Expressions

## Transportation Cost of the network (note: independent of prouduct)
## Model of these conversions ::
    ## Flow Quantity from source of destination (ton) / truck capacity (ton/truck) == Number of trucks running on the arc
    ## number of trucks * Miles * cost per mile per truck == Cost of Transportation
def convertion_to_transportation_cost1(model,src,dst):
    global plant_to_customer,cost_per_truck_per_mile,std_truck_capacity
    return cost_per_truck_per_mile*plant_to_customer[(src,dst)]["Distance"]*(sum(model.x_plant_to_customer[src,p,dst] for p in model.products)/std_truck_capacity)
model.cost_plant_to_customer = Expression(model.plants, model.customers, rule = convertion_to_transportation_cost1)

def convertion_to_transportation_cost2(model,src,dst):
    global customer_to_customer,cost_per_truck_per_mile,std_truck_capacity
    return cost_per_truck_per_mile*customer_to_customer[(src,dst)]["Distance.1"]*(sum(model.x_WH_to_customer[src,p,dst] for p in model.products)/std_truck_capacity)
model.cost_WH_to_customer = Expression(model.potential_WH, model.customers, rule = convertion_to_transportation_cost2)

def convertion_to_transportation_cost3(model,src,dst):
    global customer_to_customer,cost_per_truck_per_mile,std_truck_capacity
    return cost_per_truck_per_mile*plant_to_customer[(src,dst)]["Distance"]*(sum(model.x_plant_to_WH[src,p,dst] for p in model.products)/std_truck_capacity)
model.cost_plant_to_WH = Expression(model.plants ,model.potential_WH, rule = convertion_to_transportation_cost3)

def summation_transporation_cost(model):
    a = sum(model.cost_plant_to_customer[s,d] for s in model.plants for d in model.customers)
    b = sum(model.cost_plant_to_WH[s,d] for s in model.plants for d in model.potential_WH)
    c = sum(model.cost_WH_to_customer[s,d] for s in model.potential_WH for d in model.customers)
    return a+b+c
model.transportation_cost = Expression(rule = summation_transporation_cost)

## Production Costs
## Quantity produced (ton) * production cost per ton
def conversion_to_production_cost(model):
    return sum(model.Q[pl,pr]*model.production_cost[pl,pr] for pl in model.plants for pr in model.products)
model.total_production_cost = Expression(rule = conversion_to_production_cost)

## Changeover_Costs
## Number of changeover = Total Setup days * cost per day for setup
def conversion_to_changeover_cost(model,pl):
    number_of_setup_days = sum(model.Cp[pl,p1,p2]*model.changeover_days[p1,p2] for p1 in model.products for p2 in model.products)
    return number_of_setup_days*setup_cost_per_day
model.changeover_cost_plant = Expression(model.plants, rule=conversion_to_changeover_cost)

def summation_setup_cost(model):
    return sum(model.changeover_cost_plant[pl] for pl in model.plants)
model.total_setup_cost = Expression(rule = summation_setup_cost)

## Projected profit = Revenue - transportation_cost - production_cost - changeover_cost
def profit_equation(model):
    return model.total_revenue - (model.transportation_cost + model.total_production_cost + model.total_setup_cost)
model.total_profit = Expression(rule = profit_equation)

#################################################
"Scenario Specific Constraints"

## Scenario 1 ::

## To staisfy scenario 1 specification
## Find all the arcs with distance < 500 miles (origin can be either plant or potential warehouse)
## These all arcs will carry a flow >= 80 % of the total agreegated demand

## For new fulfillment policy, Find all the edges in the network with distance < 500
edges1 = set((pl,c) for pl in model.plants for c in model.customers  if plant_to_customer[(pl,c)]["Distance"] < km_serv_dist)
edges2 = set((w,c) for w in model.potential_WH for c in model.customers  if customer_to_customer[(c,w)]["Distance.1"] < km_serv_dist)
## possibility 1 :: Plant to customer directly (edges 1)
## possibility 2 :: potential_WH to customer  (edges 2)

def fullfillment_policy(model):
    global edges1, edges2, p_serv_level
    total_flow = sum(model.x_plant_to_customer[pl,p,c] for p in model.products for (pl,c) in edges1) + sum(model.x_WH_to_customer[w,p,c] for p in model.products for (w,c) in edges2)
    total_demand = sum(model.demand[c,p] for c in model.customers for p in model.products)
    return total_flow >= (p_serv_level/100)*total_demand
model.serviceability = Constraint(rule = fullfillment_policy)

## Constraints to force existing setup
# def force1(model,pl,pr):  ### to disengage scenario 2
#     global product_plant_assignment
#     if product_plant_assignment[pl][pr] == 0:
#         return model.yp[pl,pr] == 0
#     else:
#         return Constraint.Skip
#     # return model.yp[pl,pr] <= product_plant_assignment[pl][pr]
# model.assign_product_to_plant = Constraint(model.plants, model.products, rule = force1)

##
print ("Constraints & Expressions :: Completed")
###########

def objective_function(model):
    return (-1*(model.total_profit)/182346777) + 1000*sum(model.y_plant_to_WH[src,p,dst] for src in model.plants for p in model.products for dst in model.potential_WH)
model.obj1 = Objective(sense = minimize, rule = objective_function)

print ("Objective :: Completed")

solution = solve_model(model)

model = solution[0]
results = solution[1]

model.total_revenue.pprint()

print (results)
## KPI comparison::

## Profit Difference (along with Revenue, Transportation, Production and Changeover Costs)
## Number of trucks in operation + miles traveled
## Plant Capacity Utilization >> Pending
## Investment and Recovery
## Demand Served in the specified limits of service distance

## Profit and Other Costs
cost_projections = {
                    'profit'  :    4*round(value(model.total_profit),2),
                    'transportation_cost' : 4*round(value(model.transportation_cost),2),
                    'changeover_cost'  : 4*round(value(model.total_setup_cost),2),
                    'production_cost'  : 4*round(value(model.total_production_cost),2)
                    }

cost_breakups = {
                    'cost_shipping_plant_to_WH': 4*sum(round(value(model.cost_plant_to_WH[s,d]),2) for s in model.plants for d in model.potential_WH),
                    'cost_shipping_WH_to_customer': 4*sum(round(value(model.cost_WH_to_customer[s,d]),2) for s in model.potential_WH for d in model.customers),
                    'cost_shipping_plant_to_customer': 4*sum(round(value(model.cost_plant_to_customer[s,d]),2) for s in model.plants for d in model.customers),
                    'changeover_costs': {pl: 4*round(value(model.changeover_cost_plant[pl]),2) for pl in model.plants},
                    'production': {pl :{pr: 4*round(value(model.Q[pl,pr]),2) for pr in model.products} for pl in model.plants}
                }


t1 = sum(round(sum(value(model.x_plant_to_WH[src,p,dst]) for p in model.products)/10) for src in model.plants for dst in model.potential_WH)
t2 = sum(round(sum(value(model.x_WH_to_customer[src,p,dst]) for p in model.products)/10) for src in model.potential_WH for dst in model.customers)
t3 = sum(round(sum(value(model.x_plant_to_customer[src,p,dst]) for p in model.products)/10) for src in model.plants for dst in model.customers)

p = sum(value(model.x_plant_to_customer[pl,p,c]) for p in model.products for (pl,c) in edges1) + sum(value(model.x_WH_to_customer[w,p,c]) for p in model.products for (w,c) in edges2)
q = sum(model.demand[c,p] for c in model.customers for p in model.products)
serv_percent = round(100*(p/q),1)

capacity_utilization = {}

for pl in model.plants:
    solved_running_hours = sum(value(model.Q[pl,pr])/model.rate_of_production[pl] for pr in model.products)  ## (tonnes/q) divided by (tonnes/hrs) == hrs/q
    solved_changeover_hours = sum(w_hours*(value(model.Cp[pl,p1,p2])*model.changeover_days[p1,p2]) for p1 in model.products for p2 in model.products)   ## changeover_days * 8 hrs a day == changeover time
    total_available_hours = 3*(30*w_hours)
    p_cap = round(100*(solved_running_hours + solved_changeover_hours)/total_available_hours)
    overtime = round(max(0,(solved_running_hours+solved_changeover_hours)-total_available_hours))
    capacity_utilization[pl] = {"p_capacity":p_cap, "quarterly_overtime(hrs)":overtime}

additional_kpi = {
                    "number_of_trucks" : 4*(t1 + t2 + t3),
                    "percent_demand_within_serv_distance" : serv_percent,
                    "capacity_metrics":capacity_utilization
                }

warehouse_locations = [w for w in model.potential_WH if sum(round(value(model.y_plant_to_WH[pl,p,w]),2) for pl in model.plants for p in model.products)]

product_assignments = {pl:{pr: value(model.yp[pl,pr]) for pr in model.products} for pl in model.plants}

print ("\n\n************* KPI's SUMMARY OF OUTPUT DATASET (at annual level)*********************\n")

print ("\nCOST PROJECTIONS\n")
pprint (cost_projections)
print ("\nCOST BREAKUP AND ASSIGNMENT MATRICES\n")
pprint (cost_breakups)
print("\nPROPOSED WAREHOUSE ID\n")
pprint (warehouse_locations)
print ("\nPROPOSED PRODUCTION\n")
pprint (product_assignments)
print("\nADDITIONAL KPIs\n")
pprint (additional_kpi)

### Distribution Plan ::
first_mile = []
for pl,p,w in itertools.product(model.plants,model.products,model.potential_WH):
    first_mile.append({"from_plant":pl, "to_warehouse":w, "product":p, "quantity": 4*round(value(model.x_plant_to_WH[pl,p,w]),2)})

last_mile_1 = []
for pl,p,c in itertools.product(model.plants, model.products, model.customers):
    last_mile_1.append({"from_plant":pl, "to_customer":c, "product":p,"quantity": 4*round(value(model.x_plant_to_customer[pl,p,c]),2)})

last_mile_2 = []
for w,p,c in itertools.product(model.potential_WH, model.products, model.customers):
    last_mile_2.append({"from_warehouse":w, "to_customer":c, "product":p,"quantity":4*round(value(model.x_WH_to_customer[w,p,c]),2)})

agreegate_production = []
for pl,p in itertools.product(model.plants, model.products):
    agreegate_production.append({"plant":pl,"product":p,"quantity":4*round(value(model.Q[pl,p]),2)})

reports = {
'plant_to_WH':first_mile,
'plant_to_customer':last_mile_1,
'WH_to_customer':last_mile_2,
"production" : agreegate_production
}

# import pandas

# for k,v in reports.items():
#     report_df = pandas.DataFrame(v)
#     report_df.name = k
#     report_df.to_csv(path_or_buf = "./output_files/%s.csv"%(report_df.name),index = False)

print("\nreport files exported!")
print("\nSuccess!")
