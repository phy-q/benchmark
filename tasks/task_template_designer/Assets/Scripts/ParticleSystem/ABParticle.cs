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

[RequireComponent (typeof (SpriteRenderer))]
public class ABParticle : PhysicalBody {

	private float _lifetime;
	private bool  _isDying;

	private SpriteRenderer 	 _renderer;
	private ABParticleSystem _emitter;

	public void Create (ABParticleSystem emitter) {

		mass = 0.15f;

		_renderer = GetComponent<SpriteRenderer> ();

		_emitter = emitter;
	}
	
	// Update is called once per frame
	protected override void Update () {

		base.Update ();

		if (_emitter != null)
			applyGravity = _emitter._applyGravity;
			
		if(_isDying) {

			_lifetime += 10f * Time.deltaTime;

			Color rendColor = _renderer.color;
			rendColor.a = Mathf.Lerp(rendColor.a, 0f, 10f * Time.deltaTime);
			_renderer.color = rendColor;

			if (_lifetime >= 1f)
				_emitter.KillParticle (this);
		}
	}
		
	public void Shoot(Vector2 velocity, float lifetime, bool applyGravity, bool addNoise) {

		_isDying = false;

		this.velocity = velocity;
		this.applyGravity = applyGravity;

		Vector3 noise = Vector3.zero;

		if (addNoise) {
			
			noise = Random.onUnitSphere * 0.25f;
			noise.z = 0f;
		}

		transform.position = _emitter.transform.position + noise;

		float rot = Random.Range (0f, 360f);
		transform.rotation = Quaternion.Euler (transform.rotation.x, transform.rotation.y, rot);

		gameObject.SetActive (true);

		Color rendColor = _renderer.color;
		rendColor.a = 1f;
		_renderer.color = rendColor;

		if(lifetime > 0f)
			Invoke ("Kill", lifetime);
	}

	public void Kill() {

		_isDying = true; 
	}
}
