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

public class PhysicalBody : MonoBehaviour {

	public float    mass { get; set; }
	public bool     applyGravity { get; set; }

	public Vector2  velocity { get; set; }
	public Vector2  acceleration {get; set;}

	// Use this for initialization
	protected virtual void Start () {

	}
	
	// Update is called once per frame
	protected virtual void Update () {
	
		// Apply acceleration
		if (applyGravity)
			ApplyForce (Physics2D.gravity * Time.deltaTime);

		// Update velocity using currently acceleration
		velocity += acceleration;

		// Update position using currently velocity
		transform.position += (Vector3)velocity * Time.deltaTime;

		// Reset acceleration
		acceleration = Vector2.zero;

	}

	public void ApplyForce(Vector2 force) {

		force /= mass;
		acceleration += force;
	}
}
