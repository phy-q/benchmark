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

public class ABParallaxLayer : MonoBehaviour
{
	public float _speed = 1f;
	private Vector3 _dragOrigin;

	void Start() {
		
		_dragOrigin = ABGameWorld.Instance.GameplayCam.transform.position;
	}
		
	void LateUpdate()
	{
		Vector3 dragDistance = ABGameWorld.Instance.GameplayCam.transform.position - _dragOrigin;

		// Movement
		Vector3 movement = new Vector3(_speed * -dragDistance.x, 0, 0) * Time.fixedDeltaTime;
		transform.Translate(movement);

		_dragOrigin = ABGameWorld.Instance.GameplayCam.transform.position;
	}
}
