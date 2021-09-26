// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ABObjectPool<T> where T : MonoBehaviour {

	private List<T> _pool;

	public int PolledAmount { set; get; }

	public delegate void InitObject(T obj);

	// Use this for initialization
	public ABObjectPool (int polledAmount = 10, GameObject[] reference = null, InitObject initMethod = null) {

		_pool = new List<T> ();
		PolledAmount = polledAmount;

		GameObject poolObj = new GameObject();
		poolObj.name = typeof(T).ToString() + "Pool";

		GameObject particlesParent = GameObject.Find ("ObjectPool");

		if(particlesParent == null) {

			particlesParent = new GameObject();
			particlesParent.name = "ObjectPool";
		}
			
		poolObj.transform.SetParent (particlesParent.transform);

		for(int i = 0; i < PolledAmount; i++) {

			GameObject obj = null;
				
			if (reference == null)
				obj = new GameObject ();
			else
				obj = GameObject.Instantiate (reference[Random.Range(0, reference.Length)]);

			obj.name = "PolledObj_" + i;
			obj.transform.SetParent(poolObj.transform);
			obj.SetActive (false);

			T templ = obj.AddComponent<T>();

			if(initMethod != null) 
				initMethod (templ);

			_pool.Add (templ);
		}
	}

	public List<T> GetUsedObjects() {

		List<T> particles = new List<T> ();

		for(int i = 0; i < PolledAmount; i++) {

			if (_pool [i].gameObject.activeInHierarchy)
				particles.Add(_pool [i]);
		}

		return particles;
	}

	public T GetFreeObject() {

		for(int i = 0; i < PolledAmount; i++) {

			if (!_pool [i].gameObject.activeInHierarchy) {

				_pool [i].gameObject.SetActive(true);
				return _pool [i];
			}
		}

		return null;
	}

	public void SetObjectsParent(Transform parent) {

		_pool [0].transform.parent.SetParent (parent);
	}

	public void SetFreeObject(GameObject obj) {

		for(int i = 0; i < PolledAmount; i++) {

			if (obj.name == "PolledObj_" + i) {

				_pool [i].gameObject.SetActive(false);
				return;
			}
		}
	}

	public void FreeAllObjects() {

		for(int i = 0; i < PolledAmount; i++)
			_pool [i].gameObject.SetActive(false);
	}
}
