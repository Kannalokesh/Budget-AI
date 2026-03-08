test_queries = [
    # --- VISION & GENERAL STRATEGY ---
    {
        "question": "What has been the economic trajectory of India since the government assumed office 12 years ago?",
        "ground_truth": "India’s economic trajectory has been marked by stability, fiscal discipline, sustained growth, and moderate inflation."
    },
    {
        "question": "What is the government's 'Sankalp' mentioned in the budget?",
        "ground_truth": "The government’s ‘Sankalp’ is to focus on the poor, underprivileged, and the disadvantaged."
    },
    {
        "question": "What are the three kartavyas that inspired the Budget 2026-27?",
        "ground_truth": "The three kartavyas are: 1) To accelerate and sustain economic growth; 2) To fulfil aspirations of our people and build their capacity; and 3) To ensure every family, community, region, and sector has access to resources and opportunities."
    },
    {
        "question": "How many individuals have come out of multidimensional poverty in the last decade?",
        "ground_truth": "Close to 25 crore individuals have come out of multidimensional poverty through a decade of the Government’s sustained and reform-oriented efforts."
    },
    {
        "question": "Which threefold approach is required for a supportive ecosystem according to the budget?",
        "ground_truth": "The threefold approach requires: 1) Sustaining the momentum of structural reforms; 2) A robust and resilient financial sector; and 3) Using cutting-edge technologies like AI as force multipliers."
    },

    # --- MANUFACTURING & STRATEGIC SECTORS ---
    {
        "question": "What is the Biopharma SHAKTI strategy and its proposed outlay?",
        "ground_truth": "Biopharma SHAKTI (Strategy for Healthcare Advancement through Knowledge, Technology and Innovation) is proposed to develop India as a global Biopharma manufacturing hub with an outlay of ₹10,000 crores over the next 5 years."
    },
    {
        "question": "How many new National Institutes of Pharmaceutical Education and Research (NIPER) are proposed?",
        "ground_truth": "The strategy includes setting up 3 new National Institutes of Pharmaceutical Education and Research (NIPER) and upgrading 7 existing ones."
    },
    {
        "question": "What are the objectives of the India Semiconductor Mission (ISM) 2.0?",
        "ground_truth": "ISM 2.0 aims to produce equipment and materials, design full-stack Indian IP, fortify supply chains, and focus on industry-led research and training centres."
    },
    {
        "question": "By how much is the outlay for the Electronics Components Manufacturing Scheme being increased?",
        "ground_truth": "The outlay is being increased from ₹22,919 crore to ₹40,000 crore to capitalise on investment momentum."
    },
    {
        "question": "Which states are supported to establish dedicated Rare Earth Corridors?",
        "ground_truth": "The mineral-rich states of Odisha, Kerala, Andhra Pradesh, and Tamil Nadu are supported to establish dedicated Rare Earth Corridors."
    },
    {
        "question": "What model will be used for the 3 dedicated Chemical Parks?",
        "ground_truth": "The 3 dedicated Chemical Parks will be established through a challenge route on a cluster-based plug-and-play model."
    },
    {
        "question": "What is the purpose of establishing Hi-Tech Tool Rooms by CPSEs?",
        "ground_truth": "They are established at 2 locations as digitally enabled automated service bureaus to locally design, test, and manufacture high-precision components at scale and lower cost."
    },
    {
        "question": "What does the CIE scheme for construction equipment cover?",
        "ground_truth": "The Scheme for Enhancement of Construction and Infrastructure Equipment (CIE) covers items ranging from lifts in apartments and fire-fighting equipment to tunnel-boring equipment for metros."
    },
    {
        "question": "What is the budgetary allocation for the Scheme for Container Manufacturing?",
        "ground_truth": "A budgetary allocation of ₹10,000 crore over a 5-year period is proposed to create a globally competitive container manufacturing ecosystem."
    },

    # --- TEXTILES & MSMEs ---
    {
        "question": "What are the five sub-parts of the Integrated Programme for the Textile Sector?",
        "ground_truth": "The sub-parts are: (a) National Fibre Scheme; (b) Textile Expansion and Employment Scheme; (c) National Handloom and Handicraft programme; (d) Tex-Eco Initiative; and (e) Samarth 2.0."
    },
    {
        "question": "What is the purpose of the Mahatma Gandhi Gram Swaraj initiative?",
        "ground_truth": "It aims to strengthen khadi, handloom, and handicrafts to help in global market linkage and branding, benefiting weavers, village industries, and rural youth."
    },
    {
        "question": "How many legacy industrial clusters does the government propose to revive?",
        "ground_truth": "The government proposes to introduce a scheme to revive 200 legacy industrial clusters."
    },
    {
        "question": "What is the purpose of the ₹10,000 crore SME Growth Fund?",
        "ground_truth": "The fund is dedicated to creating future 'Champions' by providing equity support and incentivizing enterprises based on select criteria."
    },
    {
        "question": "How much top-up is proposed for the Self-Reliant India Fund?",
        "ground_truth": "A top-up of ₹2,000 crore is proposed to continue support to micro enterprises and maintain their access to risk capital."
    },
    {
        "question": "What are the proposed measures to leverage the full potential of TReDS for MSMEs?",
        "ground_truth": "Measures include: (i) mandating TReDS for CPSE purchases; (ii) credit guarantees through CGTMSE for invoice discounting; (iii) linking GeM with TReDS; and (iv) introducing TReDS receivables as asset-backed securities."
    },
    {
        "question": "Who are 'Corporate Mitras' in the context of MSMEs?",
        "ground_truth": "Corporate Mitras are a cadre of accredited para-professionals facilitated by institutions like ICAI, ICSI, and ICMAI to help MSMEs meet compliance requirements at affordable costs."
    },

    # --- INFRASTRUCTURE & LOGISTICS ---
    {
        "question": "What is the proposed Public Capex allocation for FY2026-27?",
        "ground_truth": "In FY2026-27, public capex is proposed to increase to ₹12.2 lakh crore."
    },
    {
        "question": "What is the Infrastructure Risk Guarantee Fund?",
        "ground_truth": "It is a fund proposed to provide prudently calibrated partial credit guarantees to lenders to strengthen the confidence of private developers regarding risks during construction."
    },
    {
        "question": "Which new Dedicated Freight Corridor is proposed to be established?",
        "ground_truth": "A new Dedicated Freight Corridor connecting Dankuni in the East to Surat in the West is proposed."
    },
    {
        "question": "Which National Waterway (NW) will be operationalised first in the next 5 years?",
        "ground_truth": "The operationalisation will start with NW-5 in Odisha to connect mineral-rich areas of Talcher and Angul."
    },
    {
        "question": "Where will ship repair ecosystems for inland waterways be set up?",
        "ground_truth": "Ship repair ecosystems will be set up at Varanasi and Patna."
    },
    {
        "question": "What is the goal of the Coastal Cargo Promotion Scheme?",
        "ground_truth": "The goal is to incentivise a modal shift from rail and road to increase the share of inland waterways and coastal shipping from 6% to 12% by 2047."
    },
    {
        "question": "What is the Seaplane VGF Scheme?",
        "ground_truth": "The Seaplane Viability Gap Funding (VGF) Scheme is introduced to provide incentives to indigenize manufacturing of seaplanes and support their operations."
    },
    {
        "question": "What is the proposed outlay for Carbon Capture Utilization and Storage (CCUS)?",
        "ground_truth": "An outlay of ₹20,000 crore is proposed over the next 5 years for CCUS technologies across five industrial sectors."
    },
    {
        "question": "What are City Economic Regions (CER) and their proposed allocation?",
        "ground_truth": "CERs are mapped regions based on specific growth drivers of cities. An allocation of ₹5,000 crore per CER over 5 years is proposed for implementing their plans."
    },
    {
        "question": "List the seven High-Speed Rail corridors mentioned as 'growth connectors'.",
        "ground_truth": "The corridors are: i) Mumbai-Pune, ii) Pune-Hyderabad, iii) Hyderabad-Bengaluru, iv) Hyderabad-Chennai, v) Chennai-Bengaluru, vi) Delhi-Varanasi, and vii) Varanasi-Siliguri."
    },

    # --- FINANCIAL SECTOR & TECHNOLOGY ---
    {
        "question": "What is the purpose of the 'High Level Committee on Banking for Viksit Bharat'?",
        "ground_truth": "To comprehensively review the sector and align it with India’s next phase of growth while safeguarding financial stability and consumer protection."
    },
    {
        "question": "Which NBFCs are proposed to be restructured as a first step?",
        "ground_truth": "It is proposed to restructure the Power Finance Corporation (PFC) and the Rural Electrification Corporation (REC)."
    },
    {
        "question": "What change is proposed for the investment limit for individual Persons Resident Outside India (PROI)?",
        "ground_truth": "The limit is proposed to increase from 5% to 10% for individual PROIs, with an overall investment limit for all individual PROIs increasing from 10% to 24%."
    },
    {
        "question": "What funds and missions are mentioned to support new emerging technologies?",
        "ground_truth": "The AI Mission, National Quantum Mission, Anusandhan National Research Fund, and Research, Development and Innovation Fund."
    },
    {
        "question": "What is the target global share for India in the Services Sector by 2047?",
        "ground_truth": "The government aims to make India a global leader in services with a 10% global share by 2047."
    },
    {
        "question": "What is the role of the 'Education to Employment and Enterprise' Standing Committee?",
        "ground_truth": "To recommend measures that focus on the Services Sector as a core driver, identify gaps, and assess the impact of emerging technologies like AI on jobs."
    },

    # --- HEALTH, AYUSH & EDUCATION ---
    {
        "question": "How many Allied Health Professionals (AHPs) does the government aim to add over 5 years?",
        "ground_truth": "The government aims to add 100,000 AHPs over the next 5 years through upgrading existing institutions and establishing new ones."
    },
    {
        "question": "What is the 'Care Ecosystem' proposal?",
        "ground_truth": "A strong Care Ecosystem covering geriatric and allied care services will be built, training 1.5 lakh caregivers in the coming year."
    },
    {
        "question": "What are the five Regional Medical Hubs proposed to support?",
        "ground_truth": "They are established to support States in promoting medical tourism services as integrated healthcare complexes."
    },
    {
        "question": "What facilities will be included in the Regional Medical Hubs?",
        "ground_truth": "They will have AYUSH Centres, Medical Value Tourism Facilitation Centres, and infrastructure for diagnostics, post-care, and rehabilitation."
    },
    {
        "question": "What are the three specific proposals for AYUSH development?",
        "ground_truth": "1) Set up 3 new All India Institutes of Ayurveda; 2) Upgrade AYUSH pharmacies and Drug Testing Labs; 3) Upgrade the WHO Global Traditional Medicine Centre in Jamnagar."
    },
    {
        "question": "What is the 'Orange Economy' proposal for schools and colleges?",
        "ground_truth": "Setting up AVGC (Animation, Visual Effects, Gaming and Comics) Content Creator Labs in 15,000 secondary schools and 500 colleges."
    },
    {
        "question": "Where is the new National Institute of Design proposed to be established?",
        "ground_truth": "In the eastern region of India to boost design education and development."
    },
    {
        "question": "What are 'University Townships' in the budget?",
        "ground_truth": "Five University Townships will be created near industrial corridors hosting universities, research institutions, skill centres, and residential complexes."
    },
    {
        "question": "What is the proposal for girl students in STEM institutions?",
        "ground_truth": "One girls’ hostel will be established in every district to address challenges of prolonged hours of study."
    },
    {
        "question": "Which four Telescope Infrastructure facilities are being set up or upgraded?",
        "ground_truth": "National Large Solar Telescope, National Large Optical-infrared Telescope, Himalayan Chandra Telescope, and the COSMOS-2 Planetarium."
    },

    # --- TOURISM, CULTURE & SPORTS ---
    {
        "question": "What is the pilot scheme for upskilling tourist guides?",
        "ground_truth": "Upskilling 10,000 guides in 20 iconic tourist sites through a standardized 12-week training course in hybrid mode with an IIM."
    },
    {
        "question": "What is the National Destination Digital Knowledge Grid?",
        "ground_truth": "A grid to digitally document all places of cultural, spiritual, and heritage significance to create a new ecosystem of jobs."
    },
    {
        "question": "Which regions are identified for the development of Mountain trails?",
        "ground_truth": "Himachal Pradesh, Uttarakhand, Jammu and Kashmir, Araku Valley in Eastern Ghats, and Podhigai Malai in Western Ghats."
    },
    {
        "question": "What is the International Big Cat Alliance and its 2026 event?",
        "ground_truth": "Established in 2024, India is hosting the first ever Global Big Cat Summit in 2026 with heads of governments from 95 countries."
    },
    {
        "question": "Name some of the 7 archeological sites proposed for development.",
        "ground_truth": "Lothal, Dholavira, Rakhigarhi, Adichanallur, Sarnath, Hastinapur, and Leh Palace."
    },
    {
        "question": "What is the Khelo India Mission?",
        "ground_truth": "A mission to transform the sports sector over the next decade through talent development pathways, training centres, and sports infrastructure."
    },

    # --- AGRICULTURE & RURAL DEVELOPMENT ---
    {
        "question": "What are the key focus areas for increasing farmer incomes?",
        "ground_truth": "Productivity enhancement, entrepreneurship for small/marginal farmers, empowering Divyangjan and vulnerable groups, and focusing on Purvodaya States."
    },
    {
        "question": "How many reservoirs and Amrit Sarovars are targeted for fisheries development?",
        "ground_truth": "Integrated development of 500 reservoirs and Amrit Sarovars."
    },
    {
        "question": "Which high-value crops are supported in coastal and hilly regions?",
        "ground_truth": "Coconut, sandalwood, cocoa, and cashew in coastal areas; Agar trees, almonds, walnuts, and pine nuts in hilly regions."
    },
    {
        "question": "What is the 'Coconut Promotion Scheme'?",
        "ground_truth": "A scheme to increase production and enhance productivity, including replacing old trees with new saplings in major coconut growing states."
    },
    {
        "question": "What are the premium global brands targeted for transformation by 2030?",
        "ground_truth": "Indian Cashew and Indian Cocoa."
    },
    {
        "question": "What is Bharat-VISTAAR?",
        "ground_truth": "A multilingual AI tool that integrates AgriStack portals and ICAR packages to enhance farm productivity and provide customized advisory support."
    },
    {
        "question": "What are SHE-Marts?",
        "ground_truth": "Self-Help Entrepreneur Marts: community-owned retail outlets for rural women-led enterprises within cluster level federations."
    },
    {
        "question": "What is the Divyangjan Kaushal Yojana?",
        "ground_truth": "A scheme providing industry-relevant and customized training for Divyangjans in sectors like IT, AVGC, and Hospitality."
    },
    {
        "question": "What is the 'NIMHANS-2' proposal?",
        "ground_truth": "The setting up of a second national institute for mental healthcare in North India."
    },
    {
        "question": "What is the goal of the 'Purvodaya' initiative for North-Eastern regions?",
        "ground_truth": "Development of an integrated East Coast Industrial Corridor, 5 tourism destinations, and provision of 4,000 e-buses."
    },

    # --- FISCAL & FINANCE ---
    {
        "question": "What is the vertical share of devolution recommended by the 16th Finance Commission?",
        "ground_truth": "The Government has accepted the recommendation to retain the vertical share of devolution at 41%."
    },
    {
        "question": "How much is provided as Finance Commission Grants to States for FY 2026-27?",
        "ground_truth": "₹1.4 lakh crore is provided to the States for Finance Commission Grants."
    },
    {
        "question": "What is the target debt-to-GDP ratio for the Central Government by 2030-31?",
        "ground_truth": "The target is reaching a debt-to-GDP ratio of 50±1 percent by 2030-31."
    },
    {
        "question": "What is the estimated fiscal deficit for FY 2026-27?",
        "ground_truth": "The fiscal deficit in BE 2026-27 is estimated to be 4.3 percent of GDP."
    },
    {
        "question": "What is the total expenditure estimated for 2026-27?",
        "ground_truth": "The total expenditure is estimated as ₹53.5 lakh crore."
    },

    # --- DIRECT TAXES (Part B) ---
    {
        "question": "When will the New Income Tax Act, 2025 come into effect?",
        "ground_truth": "The Income Tax Act, 2025 will come into effect from 1st April, 2026."
    },
    {
        "question": "What is the proposed tax exemption for Motor Accident Claims Tribunal awards?",
        "ground_truth": "Any interest awarded by the MACT to a natural person will be exempt from Income Tax, and TDS on this account will be removed."
    },
    {
        "question": "What are the new TCS rates for overseas tour program packages?",
        "ground_truth": "The TCS rate is reduced from 5% and 20% to a flat 2% without any stipulation of amount."
    },
    {
        "question": "What is the new TCS rate for education and medical purposes under LRS?",
        "ground_truth": "The rate is reduced from 5% to 2%."
    },
    {
        "question": "How is TDS on 'manpower services' being clarified?",
        "ground_truth": "It is brought within the definition of 'work' to be taxed at source as 'payment to contractors' at 1% or 2% only."
    },
    {
        "question": "What is the timeline for the 'FAST-DS' scheme?",
        "ground_truth": "It is a one-time 6-month foreign asset disclosure scheme for small taxpayers."
    },
    {
        "question": "What are the categories and thresholds under the FAST-DS scheme?",
        "ground_truth": "Category A: Up to 1 crore rupees for undisclosed assets. Category B: Up to 5 crore rupees for assets where income was disclosed but asset was not declared."
    },
    {
        "question": "What is the 'common order' proposal for penalty proceedings?",
        "ground_truth": "It proposes to integrate assessment and penalty proceedings into a common order to reduce the multiplicity of proceedings."
    },
    {
        "question": "How is the maximum punishment for tax offences being rationalised?",
        "ground_truth": "Maximum punishment for any offence (except repeated ones) is reduced to 2 years instead of 7 years."
    },
    {
        "question": "What is the proposed tax rate for corporate promoters in share buybacks?",
        "ground_truth": "Promoters will pay an additional buyback tax, making the effective rate 22% for corporate promoters."
    },
    {
        "question": "What are the new STT rates for Futures and Options?",
        "ground_truth": "STT on Futures is raised to 0.05%; STT on options premium/exercise is raised to 0.15%."
    },
    {
        "question": "What is the safe harbour margin proposed for Information Technology Services?",
        "ground_truth": "All segments are clubbed under a single category with a common safe harbour margin of 15.5 percent."
    },
    {
        "question": "What is the tax holiday proposal for foreign cloud service companies?",
        "ground_truth": "Foreign companies providing cloud services globally using data centres in India get a tax holiday till 2047, provided they use an Indian reseller for Indian customers."
    },
    {
        "question": "To what extent can MAT credit be set off in the new tax regime?",
        "ground_truth": "Set-off using available MAT credit is allowed to an extent of 1/4th (25%) of the tax liability."
    },

    # --- INDIRECT TAXES & CUSTOMS ---
    {
        "question": "What is the final tax rate for Minimum Alternate Tax (MAT) from April 2026?",
        "ground_truth": "MAT is proposed to be made a final tax at a reduced rate of 14 percent (down from 15 percent)."
    },
    {
        "question": "What is the new duty-free limit for imports used in processing seafood?",
        "ground_truth": "The limit is increased from 1% to 3% of the FOB value of the previous year’s export turnover."
    },
    {
        "question": "Which solar glass component is being exempted from basic customs duty?",
        "ground_truth": "Sodium antimonate import for use in manufacture of solar glass."
    },
    {
        "question": "Till which year is the customs duty exemption for Nuclear Power Projects extended?",
        "ground_truth": "It is extended till the year 2035."
    },
    {
        "question": "What are the customs duty changes for goods imported for personal use?",
        "ground_truth": "The tariff rate on all dutiable goods imported for personal use is reduced from 20 per cent to 10 per cent."
    },
    {
        "question": "How many drugs or medicines for cancer patients are getting customs duty exemption?",
        "ground_truth": "Basic customs duty is exempted on 17 drugs or medicines."
    },
    {
        "question": "What is the extension period for AEO (Authorised Economic Operator) duty deferral?",
        "ground_truth": "The duty deferral period is enhanced from 15 days to 30 days for Tier 2 and Tier 3 AEOs."
    },
    {
        "question": "How long is the validity of an advance ruling binding on Customs extended to?",
        "ground_truth": "It is extended from 3 years to 5 years."
    },
    {
        "question": "What are the rules for fish catch by Indian vessels in the EEZ?",
        "ground_truth": "Fish catch by an Indian fishing vessel in the Exclusive Economic Zone (EEZ) or on the High Seas will be made free of duty."
    },
    {
        "question": "What change is proposed for courier export value caps?",
        "ground_truth": "Complete removal of the current value cap of ₹10 lakh per consignment on courier exports."
    },
    {
        "question": "What is the new customs duty on Monazite?",
        "ground_truth": "The rate is reduced from 2.5% to Nil."
    },
    {
        "question": "Which household electronic item is getting a component duty exemption?",
        "ground_truth": "Specified parts used in the manufacture of microwave ovens."
    },
    {
        "question": "What is the new customs duty rate for Potassium hydroxide?",
        "ground_truth": "The rate is increased from Nil to 7.5%."
    },
    {
        "question": "What is the proposed NCCD rate for Chewing tobacco from May 2026?",
        "ground_truth": "The NCCD rate is revised to 60% (from 25%), with the effective duty rate maintained at 25% by notification."
    },
    {
        "question": "What is the jurisdiction extension of the Customs Act, 1962 for fishing?",
        "ground_truth": "The jurisdiction is being extended beyond territorial waters for the purpose of fishing and fishing-related activities."
    },
    {
        "question": "What is the proposed change regarding the removal of warehoused goods between custom bonded warehouses?",
        "ground_truth": "The requirement of prior permission from the proper officer for removal of warehoused goods from one custom bonded warehouse to another is proposed to be removed under Section 67."
    },
    {
        "question": "What is proposed regarding the merger of registered Non-Profit Organisations (NPOs)?",
        "ground_truth": "A new section is proposed to allow the merger of a registered NPO with another registered NPO having the same or similar objects, provided certain prescribed conditions are fulfilled."
    },
    {
        "question": "What is the proposal for the transition of biogas in Central Excise duty?",
        "ground_truth": "The entire value of biogas is proposed to be excluded while calculating the Central Excise duty payable on biogas blended CNG."
    }
]