# coding=utf-8
# Copyright 2018-2023 EvaDB
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
import os
from typing import List

from evadb.third_party.vector_stores.types import (
    FeaturePayload,
    VectorIndexQuery,
    VectorIndexQueryResult,
    VectorStore,
)
from evadb.utils.generic_utils import try_to_import_weaviate_client
# import redis
import redis
import json

import numpy as np
import pandas as pd
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

required_params = []
_weaviate_init_done = False


class WeaviateVectorStore(VectorStore):
    def __init__(self, collection_name: str, **kwargs) -> None:
        try_to_import_weaviate_client()
        global _weaviate_init_done

        self._collection_name = collection_name

        # Get the API key.
        self._api_key = kwargs.get("WEAVIATE_API_KEY")

        if not self._api_key:
            self._api_key = os.environ.get("WEAVIATE_API_KEY")

        assert (
            self._api_key
        ), "Please set your `WEAVIATE_API_KEY` using set command or environment variable (WEAVIATE_API_KEY). It can be found at the Details tab in WCS Dashboard."

        # Get the API Url.
        self._api_url = kwargs.get("WEAVIATE_API_URL")

        if not self._api_url:
            self._api_url = os.environ.get("WEAVIATE_API_URL")

        assert (
            self._api_url
        ), "Please set your `WEAVIATE_API_URL` using set command or environment variable (WEAVIATE_API_URL). It can be found at the Details tab in WCS Dashboard."

        if not _weaviate_init_done:
            # Initialize weaviate client
            import weaviate

            client = weaviate.Client(
                url=self._api_url,
                auth_client_secret=weaviate.AuthApiKey(api_key=self._api_key),
            )
            client.schema.get()

            _weaviate_init_done = True

        self._client = client

        # Test Redis connection 
        self._redis_host = ''
        self._redis_port = ''
        self._redis_password = ''

        # Get the Redis host name.
        self._redis_host = kwargs.get("REDIS_HOST")
        if not self._redis_host:
            self._redis_host = os.environ.get("REDIS_HOST")

        # Get the Redis port.
        self._redis_port = kwargs.get("REDIS_PORT")
        if not self._redis_port:
            self._redis_port = os.environ.get("REDIS_PORT")
        
        # Get the Redis password.
        self._redis_password = kwargs.get("REDIS_PASSWORD")
        if not self._redis_password:
            self._redis_password = os.environ.get("REDIS_PASSWORD")

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self._redis_host,
            port=self._redis_port,
            password=self._redis_password,
        ) 

    def create(
        self,
        vectorizer: str = "text2vec-openai",
        properties: list = None,
        module_config: dict = None,
    ):
        properties = properties or []
        module_config = module_config or {}

        collection_obj = {
            "class": self._collection_name,
            "properties": properties,
            "vectorizer": vectorizer,
            "moduleConfig": module_config,
        }

        if self._client.schema.exists(self._collection_name):
            self._client.schema.delete_class(self._collection_name)

        self._client.schema.create_class(collection_obj)

    # batch_size (int) - the (initial) size of the batch
    # num_workers (int) - the maximum number of parallel workers
    # dynamic (bool) - whether to adjust the batch_size based on items # in the batch
    def add(self, payload: List[FeaturePayload], _batch_size: int=100, 
            _dynamic: bool=True, _num_workers: int=2, **kwargs) -> None:
        # Configure batch parameters
        # Multi-threading Batch import 
        self._client.batch.configure(
            batch_size=_batch_size,
            dynamic=_dynamic,
            num_workers=_num_workers        
        )

        # Optionally specify a vector to represent each object. 
        # Otherwise, Weaviate will follow the relevant vectorizer setting
        # Optional: Replace with the actual vectors
        using_vectors = False
        using_print_progress = False
        for k, v in kwargs.items():
            if k == "vectors":
                vectors = v
                using_vectors = True
                break
            if k == "interval":
                interval = v
                using_print_progress = True
                break
        
        # Count data from large files
        global counter
        # Use Redis pipeline to minimize the round-trip times
        pipeline = self.redis_client.pipeline()
        # Process the payload in batches
        with self._client.batch as batch:
            for i, item in enumerate(payload):
                data_object = {"id": item.id, "vector": item.embedding}
                # Add the object to the batch
                batch.add_data_object(data_object, self._collection_name)
                
                # Generate a redis key and add the object to the pipeline
                redis_key = f"{self._collection_name}_{item.id}"
                pipeline.json().set(redis_key, "$", item)

                # Optional: Specify an object vector
                if using_vectors:
                    vector = vectors[i] 
                    # add reference to the object vector
                    batch.add_reference(
                        self._collection_name, item.id, "vector", vector
                    )
                # Optional: Calculate and display progress
                if using_print_progress:
                    counter += 1
                    if counter % interval == 0:
                        print(f'Imported {counter} items...')
            # Execute the batch
            res = pipeline.execute()

    def delete(self) -> None:
        self._client.schema.delete_class(self._collection_name)

    def query(self, query: VectorIndexQuery, using_cache = True) -> VectorIndexQueryResult:
        # Check Redis first
        if using_cache:
            cache_key = f"{self._collection_name}_{query.id}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                # print("Found in Redis!")
                results = json.loads(cached_result)
                similarities = [item["_additional"]["distance"] for item in results]
                ids = [item["id"] for item in results]
                return VectorIndexQueryResult(similarities, ids)
                
        # Query Weaviate if not use Redis or not in Redis
        response = (
            self._client.query.get(self._collection_name, ["*"])
            .with_near_vector({"vector": query.embedding})
            .with_limit(query.top_k)
            .do()
        )
        data = response.get("data", {})
        results = data.get("Get", {}).get(self._collection_name, [])

        # Set the cache with 3600 seconds expiration
        if using_cache:
            self.redis_client.setex(cache_key, 3600, json.dumps(response))  

        similarities = [item["_additional"]["distance"] for item in results]
        ids = [item["id"] for item in results]

        return VectorIndexQueryResult(similarities, ids)
