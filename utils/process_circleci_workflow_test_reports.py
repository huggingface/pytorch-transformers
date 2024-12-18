# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os

import requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id

    r = requests.get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = r.json()["items"]

    os.makedirs("outputs", exist_ok=True)

    workflow_summary = {}
    # for each job, download artifacts
    for job in jobs:
        project_slug = job["project_slug"]
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            url = f'https://circleci.com/api/v2/project/{project_slug}/{job["job_number"]}/artifacts'
            r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]

            os.makedirs(job["name"], exist_ok=True)
            os.makedirs(f'outputs/{job["name"]}', exist_ok=True)

            job_test_summaries = {}
            for artifact in job_artifacts:
                if artifact["path"].startswith("reports/") and artifact["path"].endswith("/summary_short.txt"):
                    node_index = artifact["node_index"]
                    url = artifact["url"]
                    r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    test_summary = r.text
                    job_test_summaries[node_index] = test_summary

            summary = {}
            for node_index, node_test_summary in job_test_summaries.items():
                for line in node_test_summary.splitlines():
                    if line.startswith("PASSED "):
                        test = line[len("PASSED ") :]
                        summary[test] = "passed"
                    elif line.startswith("FAILED "):
                        test = line[len("FAILED ") :].split()[0]
                        summary[test] = "failed"
            # failed before passed
            summary = dict(sorted(summary.items(), key=lambda x: (x[1], x[0])))
            workflow_summary[job["name"]] = summary

            # collected version
            with open(f'outputs/{job["name"]}/test_summary.json', "w") as fp:
                json.dump(summary, fp, indent=4)

    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            if test not in new_workflow_summary:
                new_workflow_summary[test] = {}
            new_workflow_summary[test][job_name] = status

    for test, result in new_workflow_summary.items():
        new_workflow_summary[test] = dict(sorted(result.items()))
    new_workflow_summary = dict(sorted(new_workflow_summary.items()))

    with open("outputs/test_summary.json", "w") as fp:
        json.dump(new_workflow_summary, fp, indent=4)
