# retirement_lifestyle_full.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import math

st.set_page_config(page_title="Retirement & Household Simulator (Full)", layout="wide")

# -------------------- Helper finance & simulation functions --------------------
def years_to_retirement(age, retirement_age):
    return max(0, retirement_age - age)

def annual_medical_cost_for_member(member_age, member_diseases, has_insurance, premium,
                                   base_visit_cost=1000, base_disease_map=None,
                                   medical_inflation=0.10, years_from_now=0):
    """
    Estimate annual medical cost for one member.
    - base_visit_cost: cost per doctor visit assumed already counted elsewhere.
    - member_diseases: list of strings.
    - medical_inflation: annual medical inflation rate used when projecting costs into future.
    - years_from_now: used when projecting cost to retirement (inflate).
    Age multiplier: costs increase after age 40 progressively.
    """
    if base_disease_map is None:
        base_disease_map = {
            "Diabetes": 20000, "Cancer": 100000, "Heart disease": 50000,
            "Kidney disease": 40000, "Lung disease": 30000, "Stomach disease": 20000
        }
    # Sum base disease costs
    disease_sum = sum(base_disease_map.get(d, 0) for d in member_diseases)
    # Age adjustment multiplier: +3% per year after 40
    age_adj = 1.0
    if member_age > 40:
        age_adj += 0.03 * (member_age - 40)
    # baseline medical spend includes some base visits (assume 2) plus disease costs
    baseline = (2 * base_visit_cost) + disease_sum
    # apply age adj
    baseline *= age_adj
    # project forward by medical inflation
    baseline *= (1 + medical_inflation) ** years_from_now
    # if insured, assume insurance covers 70% of unpredictable medical costs, but premium is added fully
    if has_insurance:
        out_of_pocket = 0.30 * baseline + premium
        return out_of_pocket
    else:
        return baseline

def total_medical_cost_for_household(members, medical_inflation, years_from_now=0):
    total = 0.0
    for m in members:
        total += annual_medical_cost_for_member(
            member_age=m["age"],
            member_diseases=m["diseases"],
            has_insurance=m["insured"],
            premium=m.get("premium", 0.0),
            medical_inflation=medical_inflation,
            years_from_now=years_from_now
        )
    return total

def annual_grocery_cost(monthly_veg, monthly_fruits, monthly_spices, monthly_oats, monthly_oils, monthly_nonveg):
    """All monthly inputs are currency amounts."""
    return 12 * (monthly_veg + monthly_fruits + monthly_spices + monthly_oats + monthly_oils + monthly_nonveg)

def annual_food_other_cost(family_size, rotis_per_person, rice_kg_per_person, restaurants_per_month, fast_food_per_week):
    roti_cost = 3
    rice_cost = 75
    restaurant_cost = 1200
    fast_food_cost = 300
    food = family_size * rotis_per_person * 365 * roti_cost
    rice = family_size * rice_kg_per_person * 365 * rice_cost
    rest = restaurants_per_month * restaurant_cost * 12
    fast = fast_food_per_week * fast_food_cost * 52
    return food + rice + rest + fast

def annual_househelp_cost(has_help, monthly_salary):
    return 12 * monthly_salary if has_help else 0.0

def annual_utilities(electricity, mobile, tv_internet, other):
    return 12 * (electricity + mobile + tv_internet + other)

def annual_emi_cost(monthly_emi):
    return 12 * monthly_emi

def sip_future_value(monthly_investment, rate_annual, years):
    if monthly_investment <= 0 or years <= 0:
        return 0.0
    r = rate_annual / 12.0
    n = years * 12
    fv = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
    return fv

def estimate_historic_mu_sigma(ticker, fallback_mu=0.08, fallback_sigma=0.15):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5y")["Close"].dropna()
        if hist.shape[0] < 50:
            return fallback_mu, fallback_sigma
        daily_ret = hist.pct_change().dropna()
        mu = daily_ret.mean() * 252
        sigma = daily_ret.std() * math.sqrt(252)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return fallback_mu, fallback_sigma
        return float(mu), float(sigma)
    except Exception:
        return fallback_mu, fallback_sigma

def annual_savings_needed(current_investments, target_corpus, years, real_return):
    if years <= 0:
        return max(0.0, target_corpus - current_investments)
    r = real_return
    fv_current = current_investments * ((1 + r) ** years)
    annuity_factor = ((1 + r) ** years - 1) / r
    return max(0.0, (target_corpus - fv_current) / annuity_factor)

def target_corpus_from_withdrawal(desired_withdrawal_amt, withdrawal_rate=0.04):
    if withdrawal_rate <= 0:
        return float("inf")
    return desired_withdrawal_amt / withdrawal_rate

def montecarlo_accumulate_and_decumulate(init_portfolio, yearly_contribution, years_accum, years_decum,
                                         expected_return, volatility, withdrawal_amount, sims=2000, seed=None):
    rng = np.random.default_rng(seed)
    sims = int(sims)
    total_years = years_accum + years_decum
    full_paths = np.zeros((sims, total_years))
    finals = np.zeros(sims)
    for s in range(sims):
        pv = init_portfolio
        path = []
        # accumulation
        for y in range(years_accum):
            ret = rng.normal(expected_return, volatility)
            pv = pv * (1 + ret) + yearly_contribution
            path.append(pv)
        # decumulation: withdraw at start, then apply return
        for y in range(years_decum):
            pv = pv - withdrawal_amount
            if pv < 0:
                pv = -1.0  # busted
            else:
                ret = rng.normal(expected_return, volatility)
                pv = pv * (1 + ret)
            path.append(pv)
        full_paths[s, :] = path
        finals[s] = pv
    percentiles = {p: np.percentile(full_paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    success_rate = np.mean(finals >= 0)
    return success_rate, finals, percentiles, full_paths

# -------------------- New helpers (Emergency fund & required-income calc) --------------------
def get_currency_symbol(country_name):
    # simple mapping; extendable
    c = country_name.strip().lower()
    mapping = {
        "india": "₹",
        "united states": "$",
        "usa": "$",
        "us": "$",
        "united kingdom": "£",
        "uk": "£",
        "europe": "€",
        "germany": "€",
        "france": "€",
        "canada": "CA$",
        "australia": "A$"
    }
    return mapping.get(c, "")  # blank if unknown

def compute_emergency_fund(annual_consumption_now, members, medical_inflation, years_from_now=0):
    """
    Compute an emergency fund that is driven by two components:
    1) base months of living expenses (6 months default; bumped to 12 months if any severe disease present)
    2) medical buffer = 2 * estimated annual household medical cost (to handle prolonged treatments)
    Final emergency fund = max(base_buffer, medical_buffer)
    """
    # check for presence of very severe disease (Cancer) in any member
    severe_present = any("Cancer" in m["diseases"] for m in members)
    base_months = 12 if severe_present else 6
    base_buffer = (annual_consumption_now / 12.0) * base_months

    # medical now:
    medical_now = total_medical_cost_for_household(members, medical_inflation, years_from_now=years_from_now)
    medical_buffer = 2.0 * medical_now  # cover ~2 years of medical emergencies

    emergency_required = max(base_buffer, medical_buffer)
    return {
        "emergency_required": emergency_required,
        "base_months": base_months,
        "base_buffer": base_buffer,
        "medical_now": medical_now,
        "medical_buffer": medical_buffer
    }

def compute_required_income_for_goals(
    annual_consumption_now,
    annual_savings_req_for_retirement,
    current_investments,
    emergency_required,
    house_price,
    car_price,
    luxury_increase_annual=84000,
    years_to_achieve=10,
    dp_share=0.20
):
    """
    Compute the annual income required so that:
     - user can save annual_savings_req_for_retirement each year
     - user can, within years_to_achieve, save for house and car downpayments and luxury increase
     - emergency fund shortfall (if any) is also built up over the same years_to_achieve
    We distribute the downpayment+luxury and emergency shortfall evenly across years_to_achieve.
    Returns required annual income and a breakdown.
    """
    house_dp = house_price * dp_share
    car_dp = car_price * dp_share
    luxury_total_10y = luxury_increase_annual * years_to_achieve

    # annual amount needed to accumulate downpayments + luxury over target horizon
    annual_downpay_and_luxury = (house_dp + car_dp + luxury_total_10y) / years_to_achieve

    # emergency shortfall needed now (if current investments < emergency_required) - spread across years_to_achieve
    emergency_shortfall = max(0.0, emergency_required - current_investments)
    annual_emergency_build = emergency_shortfall / years_to_achieve

    # required income must at least cover current consumption + luxury increase + retirement annual savings + the yearly build for downpayments and emergency
    required_income = (annual_consumption_now + luxury_increase_annual +
                       (annual_savings_req_for_retirement if annual_savings_req_for_retirement is not None else 0.0) +
                       annual_downpay_and_luxury + annual_emergency_build)

    breakdown = {
        "annual_consumption_now": annual_consumption_now,
        "luxury_increase_annual": luxury_increase_annual,
        "annual_savings_req_for_retirement": annual_savings_req_for_retirement,
        "annual_downpay_and_luxury": annual_downpay_and_luxury,
        "emergency_shortfall": emergency_shortfall,
        "annual_emergency_build": annual_emergency_build,
        "required_income": required_income,
        "house_dp": house_dp,
        "car_dp": car_dp,
        "years_to_achieve": years_to_achieve
    }
    return breakdown

# -------------------- UI Inputs --------------------
st.title("Retirement + Household + Investments Simulator (Full)")

with st.sidebar:
    st.header("Personal & Location")
    name = st.text_input("Name (optional)")
    age = st.number_input("Your age", 18, 90, 30)
    retirement_age = st.number_input("Planned retirement age", 40, 85, 60)
    retire_to_age = st.number_input("Simulate decumulation until age", retirement_age + 5, 110, 95)
    country = st.text_input("Country", "India")
    city = st.text_input("City", "Kolkata")

    st.markdown("---")
    st.header("Demographics & family projection")
    current_family_size = st.number_input("Current family size (people)", 1, 12, 3)
    projected_family_at_ret = st.number_input("Projected family size at retirement", 1, 12, max(1, current_family_size))
    projected_family_in_15y = st.number_input("Projected family size after 15 years", 1, 12, max(1, current_family_size))
    projected_family_in_30y = st.number_input("Projected family size after 30 years", 1, 12, max(1, current_family_size))

    st.markdown("---")
    st.header("Inflation & rates")
    general_inflation = st.slider("Expected annual general inflation in your country (%)", 0.0, 15.0, 6.0, 0.1) / 100.0
    medical_inflation = st.slider("Expected annual medical inflation in your country (%)", 0.0, 25.0, 10.0, 0.1) / 100.0

    st.markdown("---")
    st.header("Income & Investments")
    annual_salary = st.number_input("Annual salary / business income (₹)", 0.0, 2_000_000_000.0, 300_000.0, step=10000.0)
    current_investments = st.number_input("Current investments total (₹) — lumps across MF/Stocks/other", 0.0, 1e10, 200_000.0, step=10000.0)
    sip_monthly = st.number_input("Monthly SIP contribution (₹)", 0.0, 500_000.0, 5_000.0, step=500.0)
    sip_expected_return = st.slider("Expected SIP annual return (%)", 0.0, 25.0, 12.0, 0.1)/100.0
    mf_current = st.number_input("Mutual Funds current lump amount (₹)", 0.0, 1e9, 200_000.0, step=10000.0)
    mf_expected_return = st.slider("Expected MF annual return (%)", 0.0, 25.0, 10.0, 0.1)/100.0
    st.markdown("Stocks (comma separated tickers and amounts):")
    stocks_input = st.text_input("Tickers (Yahoo format, e.g., RELIANCE.BO, TCS.NS)", "RELIANCE.BO")
    stocks_amounts_input = st.text_input("Amounts per ticker (comma separated)", "100000")
    fallback_stock_return = st.slider("Fallback expected stock annual return (%)", 0.0, 30.0, 8.0, 0.1)/100.0

    st.markdown("---")
    st.header("Consumption - Grocery & Food (monthly amounts)")
    monthly_veg = st.number_input("Vegetables (monthly ₹)", 0.0, 200000.0, 4000.0)
    monthly_fruits = st.number_input("Fruits (monthly ₹)", 0.0, 200000.0, 2000.0)
    monthly_spices = st.number_input("Spices (monthly ₹)", 0.0, 50000.0, 500.0)
    monthly_oats = st.number_input("Oats/Cereals (monthly ₹)", 0.0, 50000.0, 500.0)
    monthly_oils = st.number_input("Oils (monthly ₹)", 0.0, 50000.0, 1500.0)
    monthly_nonveg = st.number_input("Non-veg (monthly ₹)", 0.0, 200000.0, 5000.0)

    st.markdown("---")
    st.header("Other food & habits")
    rotis_per_person = st.slider("Avg rotis per person per day", 0, 20, 4)
    rice_kg_per_person = st.slider("Rice kg per person per day", 0.0, 1.5, 0.25, 0.05)
    restaurants_per_month = st.slider("Restaurant visits per month", 0, 40, 2)
    fast_food_per_week = st.slider("Fast food meals per week", 0, 30, 2)

    st.markdown("---")
    st.header("Household & bills")
    has_help = st.checkbox("Do you employ househelp?")
    help_monthly_salary = st.number_input("Househelp monthly salary (₹)", 0.0, 200000.0, 5000.0) if has_help else 0.0
    monthly_emi = st.number_input("Total monthly EMIs (₹)", 0.0, 2_000_000.0, 15000.0)
    electricity_monthly = st.number_input("Monthly electricity bill (₹)", 0.0, 200000.0, 3000.0)
    mobile_monthly = st.number_input("Monthly mobile bill (₹)", 0.0, 50000.0, 1000.0)
    tv_internet_monthly = st.number_input("Monthly TV/Internet (₹)", 0.0, 200000.0, 1000.0)
    other_bills_monthly = st.number_input("Other monthly bills including house rent, if any (₹)", 0.0, 200000.0, 2000.0)

    st.markdown("---")
    st.header("Medical & household members")
    st.write("For each household member enter age, diseases, insurance & premium")
    members = []
    for i in range(int(current_family_size)):
        st.subheader(f"Member {i+1}")
        m_age = st.number_input(f"Age of member {i+1}", 0, 120, 30, key=f"age_{i}")
        m_diseases = st.multiselect(f"Diseases for member {i+1}",
                                    options=["Diabetes", "Cancer", "Heart disease", "Kidney disease", "Lung disease", "Stomach disease"],
                                    key=f"diseases_{i}")
        m_insured = st.checkbox(f"Medical insurance for member {i+1}?", key=f"ins_{i}")
        m_premium = st.number_input(f"Annual premium for member {i+1} (₹)", 0.0, 1_000_000.0, 0.0, key=f"prem_{i}") if m_insured else 0.0
        members.append({"age": int(m_age), "diseases": m_diseases, "insured": m_insured, "premium": float(m_premium)})

    st.markdown("---")
    st.header("Retirement & simulation choices")
    withdrawal_mode = st.radio("Express desired withdrawal after retirement as:",
                              ("Absolute amount (₹)", "Percent of pre-retirement income"))
    desired_withdrawal_amount = None
    desired_withdrawal_pct = None
    if withdrawal_mode.startswith("Absolute"):
        desired_withdrawal_amount = st.number_input("Desired annual withdrawal (₹)", 0.0, 1e10, 360_000.0)
    else:
        desired_withdrawal_pct = st.slider("Desired withdrawal as % of pre-retirement income", 0, 300, 70)/100.0

    withdrawal_rate = st.slider("Withdrawal rate (%) for corpus calc", 2.0, 6.0, 4.0, 0.1)/100.0
    sims = st.number_input("Monte Carlo simulations", 100, 5000, 2000, step=100)
    assumed_vol = st.slider("Assumed portfolio volatility (%) if unable to infer", 5.0, 60.0, 15.0, 0.5)/100.0
    st.markdown("---")
    st.button("Recalculate / Run Simulations below")

# -------------------- Derived Investment calculations --------------------
years_left = years_to_retirement(age, retirement_age)

# parse stocks
tickers = [t.strip() for t in stocks_input.split(",") if t.strip() != ""]
try:
    stock_amounts = [float(x.strip()) for x in stocks_amounts_input.split(",")]
except:
    # default amounts if parse fails
    stock_amounts = [0.0] * len(tickers)
if len(stock_amounts) < len(tickers):
    stock_amounts += [0.0] * (len(tickers) - len(stock_amounts))

# estimate stocks mu/sigma and project naive future using mu
total_stock_current = sum(stock_amounts)
total_stock_future = 0.0
stocks_mu_sigma = {}
for i, tk in enumerate(tickers):
    amt = stock_amounts[i] if i < len(stock_amounts) else 0.0
    mu, sigma = estimate_historic_mu_sigma(tk, fallback_mu=fallback_stock_return, fallback_sigma=assumed_vol)
    stocks_mu_sigma[tk] = {"mu": mu, "sigma": sigma, "amount": amt}
    total_stock_future += amt * ((1 + mu) ** years_left)

# SIP & MF projections
sip_fv = sip_future_value(sip_monthly, sip_expected_return, years_left)
mf_future = mf_current * ((1 + mf_expected_return) ** years_left)

projected_investments_at_ret = current_investments + sip_fv + mf_future + total_stock_future

# crude investment income estimate per year (average)
if years_left > 0:
    investment_income_est = (sip_fv + mf_future + total_stock_future - (sip_monthly*12*years_left + mf_current + total_stock_current)) / max(1, years_left)
else:
    investment_income_est = 0.0

# -------------------- Consumption calculations (current & projected) --------------------
# current medical cost (now)
medical_cost_now = total_medical_cost_for_household(members, medical_inflation, years_from_now=0)

# food/grocery current annual
grocery_annual = annual_grocery_cost(monthly_veg, monthly_fruits, monthly_spices, monthly_oats, monthly_oils, monthly_nonveg)
food_other_annual = annual_food_other_cost(current_family_size, rotis_per_person, rice_kg_per_person, restaurants_per_month, fast_food_per_week)
househelp_annual = annual_househelp_cost(has_help, help_monthly_salary)
emi_annual = annual_emi_cost(monthly_emi)
utilities_annual = annual_utilities(electricity_monthly, mobile_monthly, tv_internet_monthly, other_bills_monthly)

annual_consumption_now = medical_cost_now + grocery_annual + food_other_annual + househelp_annual + emi_annual + utilities_annual

# Project consumption to retirement (inflating general items by general inflation, medical by medical inflation and age)
years_to_ret = years_left
# Estimate family size at retirement (use projected_family_at_ret)
family_size_at_ret = int(projected_family_at_ret)

# Project per-component:
def inflate_amount(amount, inflation_rate, years):
    return amount * ((1 + inflation_rate) ** years)

# Project medical at retirement: we need to age each member by years_to_ret and recompute medical cost for that future age, using family composition at retirement.
# Build members_at_ret: extend/truncate list to family_size_at_ret using average age increment logic if needed
members_at_ret = []
for i in range(family_size_at_ret):
    if i < len(members):
        m = members[i].copy()
        m["age"] = m["age"] + years_to_ret
        members_at_ret.append(m)
    else:
        # new/future member: assume adult at retirement? We'll add a neutral member with no disease age 30+years_to_ret? Safer to assume no diseases and age 30 + years_to_ret
        members_at_ret.append({"age": 30 + years_to_ret, "diseases": [], "insured": False, "premium": 0.0})

medical_cost_ret = total_medical_cost_for_household(members_at_ret, medical_inflation, years_from_now=0)  # function already applies age multiplier + medical inflation if years_from_now provided; we've shifted member ages so don't double-inflate

# Project grocery & food to retirement scaling by family size ratio and general inflation
grocery_annual_ret = inflate_amount(grocery_annual * (family_size_at_ret / max(1, current_family_size)), general_inflation, years_to_ret)
food_other_annual_ret = inflate_amount(food_other_annual * (family_size_at_ret / max(1, current_family_size)), general_inflation, years_to_ret)
househelp_annual_ret = inflate_amount(househelp_annual, general_inflation, years_to_ret)
emi_annual_ret = inflate_amount(emi_annual, general_inflation, years_to_ret)  # EMIs may change but we inflate
utilities_annual_ret = inflate_amount(utilities_annual, general_inflation, years_to_ret)

annual_consumption_at_retirement = medical_cost_ret + grocery_annual_ret + food_other_annual_ret + househelp_annual_ret + emi_annual_ret + utilities_annual_ret

# Also project consumption at +15 years and +30 years horizons if needed
def projected_consumption_in_years(years_future, family_size_future):
    # create members_future by aging current members by years_future and extend/truncate to family_size_future
    members_future = []
    for i in range(family_size_future):
        if i < len(members):
            m = members[i].copy()
            m["age"] = m["age"] + years_future
            members_future.append(m)
        else:
            members_future.append({"age": 30 + years_future, "diseases": [], "insured": False, "premium": 0.0})
    medical_future = total_medical_cost_for_household(members_future, medical_inflation, years_from_now=0)
    grocery_future = inflate_amount(grocery_annual * (family_size_future / max(1, current_family_size)), general_inflation, years_future)
    food_other_future = inflate_amount(food_other_annual * (family_size_future / max(1, current_family_size)), general_inflation, years_future)
    hh_future = inflate_amount(househelp_annual, general_inflation, years_future)
    emi_future = inflate_amount(emi_annual, general_inflation, years_future)
    util_future = inflate_amount(utilities_annual, general_inflation, years_future)
    total_future = medical_future + grocery_future + food_other_future + hh_future + emi_future + util_future
    return {
        "medical": medical_future,
        "grocery": grocery_future,
        "food_other": food_other_future,
        "househelp": hh_future,
        "emi": emi_future,
        "utilities": util_future,
        "total": total_future
    }

proj_15y = projected_consumption_in_years(15, int(projected_family_in_15y))
proj_30y = projected_consumption_in_years(30, int(projected_family_in_30y))

# -------------------- Retirement corpus & savings needs --------------------
# determine desired withdrawal amount
pre_ret_income_est = annual_salary + investment_income_est if 'investment_income_est' in locals() else annual_salary  # will recompute below
# compute investment_income_est (recompute with variables available)
# Use previously computed sip_fv etc to estimate investment income per year
if years_left > 0:
    investment_income_est = (sip_fv + mf_future + total_stock_future - (sip_monthly*12*years_left + mf_current + total_stock_current)) / max(1, years_left)
else:
    investment_income_est = 0.0
total_income_est = annual_salary + investment_income_est
pre_ret_income_est = total_income_est

if desired_withdrawal_amount is None and desired_withdrawal_pct is not None:
    desired_withdrawal_amount = desired_withdrawal_pct * pre_ret_income_est

# But a realistic desired withdrawal might be based on projected consumption at retirement:
# If user hasn't specified a high enough desired_withdrawal_amount, we show both: desired and consumption-derived.
desired_withdrawal_based_on_consumption = annual_consumption_at_retirement

target_corpus_user = target_corpus_from_withdrawal(desired_withdrawal_amount, withdrawal_rate) if desired_withdrawal_amount else None
target_corpus_consumption = target_corpus_from_withdrawal(desired_withdrawal_based_on_consumption, withdrawal_rate)

# Annual savings required (deterministic) to reach corpus using SIP_expected_return as proxy for real return
real_return_proxy = sip_expected_return if sip_expected_return > 0 else 0.05
annual_savings_req_user = annual_savings_needed(current_investments, target_corpus_user, years_left, real_return_proxy) if target_corpus_user else None
annual_savings_req_consumption = annual_savings_needed(current_investments, target_corpus_consumption, years_left, real_return_proxy)

# Surplus available now per year
total_income_now = annual_salary + investment_income_est
net_surplus_now = max(0.0, total_income_now - annual_consumption_now)

# -------------------- New: Emergency fund computation --------------------
emergency_info = compute_emergency_fund(annual_consumption_now, members, medical_inflation, years_from_now=0)
emergency_required = emergency_info["emergency_required"]
emergency_shortfall_now = max(0.0, emergency_required - current_investments)

# -------------------- New: Required income to meet retirement + 10-year house/car/luxury goal --------------------
# Default price assumptions (you can change these or add inputs if you want)
default_house_price = 5_000_000   # 50 lakh
default_car_price = 800_000     # 8 lakh
luxury_increase_annual = 84000   # ₹84K extra annual spending as specified
years_to_achieve = 10

required_income_breakdown = compute_required_income_for_goals(
    annual_consumption_now=annual_consumption_now,
    annual_savings_req_for_retirement=annual_savings_req_consumption,
    current_investments=current_investments,
    emergency_required=emergency_required,
    house_price=default_house_price,
    car_price=default_car_price,
    luxury_increase_annual=luxury_increase_annual,
    years_to_achieve=years_to_achieve,
    dp_share=0.20
)
required_income_to_meet_goals = required_income_breakdown["required_income"]
required_income_gap = max(0.0, required_income_to_meet_goals - total_income_now)

# Lifestyle improvement: income needed for 50% more consumption at retirement
required_income_for_50pct = 1.5 * annual_consumption_at_retirement

def lifestyle_affordability(surplus_per_year, income, consumption, years_left):
    house_price = 5_000_000
    car_price = 800_000
    dp_share = 0.20
    house_dp = house_price * dp_share
    car_dp = car_price * dp_share
    if surplus_per_year <= 0:
        return {"house": "Not affordable (no surplus)", "car": "Not affordable (no surplus)", "trips": "0"}
    years_house = house_dp / surplus_per_year
    years_car = car_dp / surplus_per_year
    trips_dom = int(surplus_per_year // 30000)
    trips_int = int(surplus_per_year // 120000)
    return {
        "house": f"In ~{int(years_house)} years" if years_house < years_left else "Not affordable before retirement",
        "car": f"In ~{int(years_car)} years" if years_car < years_left else "Not affordable before retirement",
        "trips": f"{trips_dom} domestic / {trips_int} international per year"
    }

afford_info = lifestyle_affordability(net_surplus_now, total_income_now, annual_consumption_now, years_left)

# -------------------- Simulation setup --------------------
# Prepare portfolio expected return & volatility estimate from components
total_current = current_investments + sip_monthly*12*years_left + mf_current + total_stock_current
if total_current <= 0:
    portfolio_mu = sip_expected_return
    portfolio_sigma = assumed_vol
else:
    w_sip = (sip_monthly*12*years_left) / total_current if total_current>0 else 0
    w_mf = mf_current / total_current
    w_stock = total_stock_current / total_current
    avg_stock_mu = np.mean([v["mu"] for v in stocks_mu_sigma.values()]) if len(stocks_mu_sigma)>0 else fallback_stock_return
    avg_stock_sigma = np.mean([v["sigma"] for v in stocks_mu_sigma.values()]) if len(stocks_mu_sigma)>0 else assumed_vol
    portfolio_mu = w_sip * sip_expected_return + w_mf * mf_expected_return + w_stock * avg_stock_mu
    portfolio_sigma = max(0.05, w_sip * 0.10 + w_mf * 0.12 + w_stock * avg_stock_sigma)

# desired withdrawal for MC: prefer user's desired amount if provided else consumption-based
mc_withdrawal = desired_withdrawal_amount if desired_withdrawal_amount else desired_withdrawal_based_on_consumption

# Simulation button and run
st.header("Overview & Results")
currency_symbol = get_currency_symbol(country)

colA, colB = st.columns(2)

with colA:
    st.subheader("Key current numbers")
    st.write(f"Location: {city}, {country} ({currency_symbol or 'local currency'})")
    st.metric("Current family size", f"{current_family_size}")
    st.metric("Annual consumption (now)", f"{currency_symbol}{annual_consumption_now:,.0f}")
    st.metric("Total income estimate (now)", f"{currency_symbol}{total_income_now:,.0f}")
    st.metric("Net yearly surplus", f"{currency_symbol}{net_surplus_now:,.0f}")

with colB:
    st.subheader("Retirement & corpus")
    st.metric("Years until retirement", f"{years_left}")
    st.metric("Projected consumption at retirement (yr)", f"{currency_symbol}{annual_consumption_at_retirement:,.0f}")
    st.metric("Consumption-based corpus", f"{currency_symbol}{target_corpus_consumption:,.0f}")
    if target_corpus_user:
        st.metric("User-desired corpus", f"{currency_symbol}{target_corpus_user:,.0f}")
    st.metric("Naive projected investments at retirement", f"{currency_symbol}{projected_investments_at_ret:,.0f}")

st.markdown("---")

# Monte Carlo run
if st.button("Run Monte Carlo simulation (accumulate + decumulate)"):
    years_accum = years_left
    years_decum = retire_to_age - retirement_age if retire_to_age > retirement_age else 20
    init_portfolio = current_investments
    yearly_contrib = net_surplus_now if net_surplus_now > 0 else 0.0
    success_rate, finals, percentiles, full_paths = montecarlo_accumulate_and_decumulate(
        init_portfolio=init_portfolio,
        yearly_contribution=yearly_contrib,
        years_accum=years_accum,
        years_decum=years_decum,
        expected_return=portfolio_mu,
        volatility=portfolio_sigma,
        withdrawal_amount=mc_withdrawal,
        sims=int(sims),
        seed=42
    )
    st.subheader("Monte Carlo Summary")
    st.write(f"Sims: {int(sims)} | Accum years: {years_accum} | Decum years: {years_decum}")
    st.write(f"Assumed portfolio mu: {portfolio_mu*100:.2f}%, sigma: {portfolio_sigma*100:.2f}%")
    st.metric("Probability of not depleting portfolio by age " + str(retire_to_age), f"{success_rate*100:.2f}%")
    st.write("Final portfolio distribution summary (sample):")
    st.write(pd.Series(finals).describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).to_frame().T)

    # Plot percentile bands
    years_axis = list(range(1, years_accum + years_decum + 1))
    plt.figure(figsize=(10,5))
    plt.fill_between(years_axis, percentiles[5], percentiles[95], alpha=0.2, label="5-95 pct")
    plt.fill_between(years_axis, percentiles[25], percentiles[75], alpha=0.3, label="25-75 pct")
    plt.plot(years_axis, percentiles[50], label="Median", linewidth=2)
    plt.axvline(years_accum, color="k", linestyle="--", label="Retirement")
    plt.xlabel("Years from today")
    plt.ylabel("Portfolio value (" + (currency_symbol or "") + ")")
    plt.title("Monte Carlo percentile bands")
    plt.legend()
    st.pyplot(plt.gcf())

    # Sample paths
    sample_idxs = np.random.choice(full_paths.shape[0], size=min(20, full_paths.shape[0]), replace=False)
    plt.figure(figsize=(10,5))
    for idx in sample_idxs:
        plt.plot(years_axis, full_paths[idx, :], alpha=0.8)
    plt.axvline(years_accum, color="k", linestyle="--")
    plt.xlabel("Years from today")
    plt.ylabel("Portfolio value (" + (currency_symbol or "") + ")")
    plt.title("Sample Monte Carlo portfolio paths")
    st.pyplot(plt.gcf())

# -------------------- Detailed outputs & breakdown --------------------
st.markdown("---")
st.subheader("Detailed Consumption & Projections")
tabs = st.tabs(["Now", "At Retirement", "+15 years", "+30 years", "Investments"])

with tabs[0]:
    st.write("### Current annual consumption breakdown")
    st.write(f"- Medical (now): {currency_symbol}{medical_cost_now:,.0f}")
    st.write(f"- Grocery (annual): {currency_symbol}{grocery_annual:,.0f}")
    st.write(f"- Food other (roti/rice/rest/fast): {currency_symbol}{food_other_annual:,.0f}")
    st.write(f"- Househelp: {currency_symbol}{househelp_annual:,.0f}")
    st.write(f"- EMIs: {currency_symbol}{emi_annual:,.0f}")
    st.write(f"- Utilities: {currency_symbol}{utilities_annual:,.0f}")
    st.metric("Total annual consumption (now)", f"{currency_symbol}{annual_consumption_now:,.0f}")

with tabs[1]:
    st.write("### Estimated consumption at retirement (annual)")
    st.write(f"- Medical (ret): {currency_symbol}{medical_cost_ret:,.0f}")
    st.write(f"- Grocery (ret): {currency_symbol}{grocery_annual_ret:,.0f}")
    st.write(f"- Food other (ret): {currency_symbol}{food_other_annual_ret:,.0f}")
    st.write(f"- Househelp (ret): {currency_symbol}{househelp_annual_ret:,.0f}")
    st.write(f"- EMIs (ret): {currency_symbol}{emi_annual_ret:,.0f}")
    st.write(f"- Utilities (ret): {currency_symbol}{utilities_annual_ret:,.0f}")
    st.metric("Total annual consumption (retirement)", f"{currency_symbol}{annual_consumption_at_retirement:,.0f}")

with tabs[2]:
    st.write("### Projected consumption in +15 years")
    st.write(f"- Total (15y): {currency_symbol}{proj_15y['total']:,.0f}")
    st.write(f"- Medical (15y): {currency_symbol}{proj_15y['medical']:,.0f}")
    st.write(f"- Grocery (15y): {currency_symbol}{proj_15y['grocery']:,.0f}")

with tabs[3]:
    st.write("### Projected consumption in +30 years")
    st.write(f"- Total (30y): {currency_symbol}{proj_30y['total']:,.0f}")
    st.write(f"- Medical (30y): {currency_symbol}{proj_30y['medical']:,.0f}")
    st.write(f"- Grocery (30y): {currency_symbol}{proj_30y['grocery']:,.0f}")

with tabs[4]:
    st.write("### Investments & projections")
    st.write(f"- Current investments total: {currency_symbol}{current_investments:,.0f}")
    st.write(f"- SIP future value (to retirement): {currency_symbol}{sip_fv:,.0f}")
    st.write(f"- MF future value: {currency_symbol}{mf_future:,.0f}")
    st.write(f"- Stocks (projected naive): {currency_symbol}{total_stock_future:,.0f}")
    st.metric("Projected total investments at retirement (naive)", f"{currency_symbol}{projected_investments_at_ret:,.0f}")

st.markdown("---")
st.subheader("Lifestyle improvement & affordability")
st.write(f"- Required income to improve lifestyle by 50% at retirement (annual consumption *1.5): {currency_symbol}{required_income_for_50pct:,.0f}")
for k, v in afford_info.items():
    st.write(f"- {k.capitalize()}: {v}")

# -------------------- New: Emergency fund & required-income summary --------------------
st.markdown("---")
st.subheader("Emergency Fund (health-driven)")

st.write(f"- Estimated annual medical expense now (household): {currency_symbol}{emergency_info['medical_now']:,.0f}")
st.write(f"- Medical buffer suggested (2x medical expenses): {currency_symbol}{emergency_info['medical_buffer']:,.0f}")
st.write(f"- Base buffer ({emergency_info['base_months']} months of living expenses): {currency_symbol}{emergency_info['base_buffer']:,.0f}")
st.metric("Emergency fund recommended", f"{currency_symbol}{emergency_required:,.0f}")
if emergency_shortfall_now > 0:
    st.warning(f"Emergency fund shortfall vs current investments: {currency_symbol}{emergency_shortfall_now:,.0f}")
else:
    st.success("You currently have at least the recommended emergency fund in investments.")

st.markdown("---")
st.subheader(f"Required annual income to meet retirement + 10-year goals (house + car + ₹{luxury_increase_annual:,} luxury increase)")

st.write("- Assumptions:")
st.write(f"  - 3BHK price assumed: {currency_symbol}{default_house_price:,.0f} (downpayment {20}% = {currency_symbol}{required_income_breakdown['house_dp']:,.0f})")
st.write(f"  - Car price assumed: {currency_symbol}{default_car_price:,.0f} (downpayment {20}% = {currency_symbol}{required_income_breakdown['car_dp']:,.0f})")
st.write(f"  - Luxury increase: {currency_symbol}{luxury_increase_annual:,.0f} per year (total over {years_to_achieve} years: {currency_symbol}{luxury_increase_annual*years_to_achieve:,.0f})")
st.write(f"  - Emergency shortfall (if any) will be accumulated across {years_to_achieve} years: {currency_symbol}{required_income_breakdown['annual_emergency_build']:,.0f} per year")

st.metric("Required annual income (estimate)", f"{currency_symbol}{required_income_to_meet_goals:,.0f}")
if required_income_gap > 0:
    st.warning(f"You need approx {currency_symbol}{required_income_gap:,.0f} more per year (≈ {required_income_gap/total_income_now*100:.1f}% increase) to meet retirement savings and the 10-year house/car/luxury goals, while also building any emergency shortfall.")
else:
    st.success("Your current income is sufficient (by this estimate) to meet retirement savings and the 10-year house/car/luxury goals under the given assumptions.")

# -------------------- Export summary --------------------
if st.button("Download full summary CSV"):
    summary = {
        "timestamp": [datetime.utcnow().isoformat()],
        "name": [name],
        "age": [age],
        "retirement_age": [retirement_age],
        "annual_salary": [annual_salary],
        "current_investments": [current_investments],
        "sip_monthly": [sip_monthly],
        "mf_current": [mf_current],
        "stocks_tickers": [stocks_input],
        "stocks_amounts": [stocks_amounts_input],
        "annual_consumption_now": [annual_consumption_now],
        "annual_consumption_retirement": [annual_consumption_at_retirement],
        "total_income_now": [total_income_now],
        "net_surplus_now": [net_surplus_now],
        "target_corpus_consumption": [target_corpus_consumption],
        "annual_savings_req_consumption": [annual_savings_req_consumption],
        "emergency_required": [emergency_required],
        "emergency_shortfall_now": [emergency_shortfall_now],
        "required_income_to_meet_goals": [required_income_to_meet_goals]
    }
    df = pd.DataFrame(summary)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="full_planner_summary.csv", mime="text/csv")

st.caption("This tool produces estimates. Assumptions: medical inflation may exceed general inflation; Monte Carlo uses normal approx for yearly returns. For personalized financial planning consult a certified advisor.")