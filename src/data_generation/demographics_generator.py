import pandas as pd
from faker import Faker
import random

fake = Faker()

def generate_patient_data(num_records = 100):
    blood_groups = ["A+","A-","B+","B-","AB+","Ab-","O+","O-"]

    data = []

    for i in range(1, num_records + 1):
        patient = {
            "patient_id" : i,
            "name" : fake.name(),
            "age" : random.randint(1,90),
            "gender" : random.choice(["Male","Female", "Other"]),
            "blood_group" : random.choice(blood_groups),
            "contact_number" : fake.phone_number(),
            "address" : fake.address().replace("/n","")
        }

        data.append(patient)

    df = pd.DataFrame(data)

    return df

if __name__  == "__main__":
    df = generate_patient_data(200)
    df.to_csv("data/raw/patient_demographic.csv",index=False)
    print("Synthetic data saved at patient_demographic.csv")
    