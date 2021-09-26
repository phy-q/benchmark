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

using UnityEngine;
using System.Collections;

public class ABEggTNT : ABBlock
{


	public float _explosionArea = 1f;
	public float _explosionPower = 1f;
	public float _explosionDamage = 1f;
	private bool _exploded = false;

	public override void Die(bool withEffect = true)
	{
		//ScoreHud.Instance.SpawnScorePoint(200, transform.position);
		if (!_exploded)
		{
			_exploded = true;
			Explode(transform.position, _explosionArea, _explosionPower, _explosionDamage, gameObject);
		}

		base.Die(withEffect);
	}

	public static void Explode(Vector2 position, float explosionArea, float explosionPower, float explosionDamage, GameObject explosive)
	{

		Collider2D[] colliders = Physics2D.OverlapCircleAll(position, explosionArea);

		foreach (Collider2D coll in colliders)
		{

			if (coll.attachedRigidbody && coll.gameObject != explosive && coll.GetComponent<ABBird>() == null)
			{

				float distance = Vector2.Distance((Vector2)coll.transform.position, position);
				Vector2 direction = ((Vector2)coll.transform.position - position).normalized;

				ABGameObject abGameObj = coll.gameObject.GetComponent<ABGameObject>();
				if (abGameObj)
					coll.gameObject.GetComponent<ABGameObject>().DealDamage(explosionDamage / distance);

				coll.attachedRigidbody.AddForce(direction * (explosionPower / (distance * 2f)), ForceMode2D.Impulse);
			}
		}

	}
}
